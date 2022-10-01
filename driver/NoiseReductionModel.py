import sys
from xmlrpc.client import boolean
import time
import math
import torch
import torch.nn as nn
import torch.optim.lr_scheduler
from torch.utils.data import Dataset
import random
import argparse
from driver.Config import *
from driver.SAHelper import *
from data.Dataloader import *
from data.GetNoiseData import *
from tqdm import tqdm
from sklearn import metrics
from transformers import AdamW, get_linear_schedule_with_warmup, BertConfig, BertForSequenceClassification, BertTokenizer
import wandb
import higher
import os


def rpad(array, n=300):
    """Right padding."""
    current_len = len(array)
    if current_len > n:
        return array[: n]
    extra = n - current_len
    return array + ([0] * extra)

def compute_accuracy(logits, true_tags):
        b, l = logits.size()
        train_acc = 0.0
        pred_tags = torch.argmax(logits, axis=1)
        # tag_correct = pred_tags.eq(true_tags).sum()
        train_acc += (pred_tags == true_tags).sum().item()
        return train_acc, b

class SentimentDataset(Dataset):
    def __init__(self, data, split="train"):
        """Initializes the dataset with given configuration.
        Args:
            split: str
                Dataset split, one of [train, val, test]
            root: bool
                If true, only use root nodes. Else, use all nodes.
            binary: bool
                If true, use binary labels. Else, use fine-grained.
        """
        self.sst = data

        self.data = [
            (rpad(tokenizer.encode(item[1]), n=66), int(item[0]))
            for item in self.sst
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X, y = self.data[index]
        X = torch.tensor(X)
        y = torch.tensor(y)
        return X, y

def cross_entropy(data, dev_data, test_data, classifier, optimizer_model, noise_type, noise_ratio, config, computing_loss, model_name, times):
    train_loader = torch.utils.data.DataLoader(
        data, batch_size=config.train_batch_size, shuffle=True
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_data, batch_size=config.train_batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=config.train_batch_size, shuffle=False
    )
    global_step = 0
    best_f1 = 0
    best_val_model_test_f1 = 0
    batch_num = int(np.ceil(len(data) / float(config.train_batch_size)))
    train_acc_list,train_loss_list, dev_list, dev_loss_list, test_list, test_loss_list = [], [], [], [], [], []

    
    for iter in range(config.train_iters):
        start_time = time.time()
        print('Iteration: ' + str(iter) + ', total batch num: ' + str(batch_num))
        batch_iter = 0

        correct_num, total_num, loss_value = 0.0, 0.0, 0.0
        classifier.train()
        for batch, labels in tqdm(train_loader):
            # bert_inputs, tags, masks = \
            #     batch_data_variable(onebatch, vocab)
            if noise_type != 'none':
                batch = torch.stack(batch, dim = 1)
            batch, labels = batch.to("cuda"), labels.to("cuda")
            optimizer_model.zero_grad()
            output = classifier(batch, labels=labels)
            logits = output.logits
            loss = computing_loss(logits, labels)
            loss.backward()
            optimizer_model.step()
            loss = loss / batch_num
            loss_value += loss.item()

            cur_correct, cur_count = compute_accuracy(logits, labels)
            correct_num += cur_correct
            total_num += cur_count
            acc = correct_num * 100.0 / total_num

            batch_iter += 1

        during_time = float(time.time() - start_time)
        print("Step:%d, ACC:%.2f, Iter:%d, batch:%d, time:%.2f, loss:%.2f" \
                % (global_step, acc, iter, batch_iter-1, during_time, loss_value))
        train_acc_list.append(acc)
        train_loss_list.append(loss_value)
        

        global_step += 1

        tag_correct, tag_total, dev_tag_acc, dev_loss, dev_f1= \
            evaluate(dev_loader, classifier, config.dev_file + '.' + str(global_step), dev_data)
        print("Dev: f1 = %d/%d = %.2f" % (tag_correct, tag_total, dev_f1))
        dev_list.append(dev_f1)
        dev_loss_list.append(dev_loss)

        tag_correct, tag_total, test_tag_acc, test_loss, test_f1 = \
            evaluate(test_loader, classifier, config.test_file + '.' + str(global_step), test_data)
        print("Test: f1 = %d/%d = %.2f" % (tag_correct, tag_total, test_f1))
        test_list.append(test_f1)
        test_loss_list.append(test_loss)

        if dev_f1 > best_f1:
            print("Exceed best f1: history = %.2f, current = %.2f" %(best_f1, dev_f1))
            best_f1 = dev_f1
            best_val_model_test_f1 = test_f1
            if config.save_after > 0 and iter > config.save_after:
                torch.save(classifier.state_dict(), config.save_model_path)
        print("Current best results: dev = %.2f,test = %.2f"%(best_f1, best_val_model_test_f1))

def coteaching(data, dev_data, test_data, classifier1, classifier2, optimizer_model1, optimizer_model2, noise_type, noise_ratio, config, computing_loss, model_name, times):
    train_loader = torch.utils.data.DataLoader(
        data, batch_size=config.train_batch_size, shuffle=True
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_data, batch_size=config.train_batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=config.train_batch_size, shuffle=False
    )
    global_step = 0
    best_f1 = 0
    best_val_model_test_f1 = 0
    batch_num = int(np.ceil(len(data) / float(config.train_batch_size)))
    train_acc_list1,train_loss_list1, dev_list1, test_list1 = [], [], [], []
    train_acc_list2,train_loss_list2, dev_list2, test_list2 = [], [], [], []

    forget_rate = noise_ratio
    num_graduals = 10
    exponent = 1

    forget_rates = np.ones(config.train_iters)*forget_rate
    forget_rates[:num_graduals] = np.linspace(0, forget_rate**exponent, num_graduals)    

    
    for iter in range(config.train_iters):
        start_time = time.time()
        print('Iteration: ' + str(iter) + ', total batch num: ' + str(batch_num))

        correct_num1, total_num1, correct_num2, total_num2, loss1_value, loss2_value = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        remember_rate = 1 - forget_rates[iter]

        for batch_idx,(batch, labels) in tqdm(enumerate(train_loader)):
            # bert_inputs, tags, masks = \
            #     batch_data_variable(onebatch, vocab)
            if noise_type != 'none':
                batch = torch.stack(batch, dim = 1)
            batch, labels = batch.to("cuda"), labels.to("cuda")
            num_remember = int(remember_rate * len(labels))

            with torch.no_grad():
                # select samples based on model 1
                classifier1.eval()
                y_pred1 = F.softmax(classifier1(batch)[0], dim=1)
                cross_entropy = F.cross_entropy(y_pred1, labels, reduction='none')
                batch_idx1= np.argsort(cross_entropy.cpu().numpy())[:num_remember]
                # select samples based on model 2
                classifier2.eval()
                y_pred2 = F.softmax(classifier2(batch)[0], dim=1)
                cross_entropy = F.cross_entropy(y_pred2, labels, reduction='none')
                batch_idx2 = np.argsort(cross_entropy.cpu().numpy())[:num_remember]

            #train net1
            classifier1.train()
            optimizer_model1.zero_grad()
            output = classifier1(batch[batch_idx2,:], labels = labels[batch_idx2])
            logits = output.logits
            loss1 = computing_loss(logits, labels[batch_idx2])
            loss1.backward()
            optimizer_model1.step()
            loss1 = loss1 / batch_num
            loss1_value += loss1.item()

            cur_correct1, cur_count1 = compute_accuracy(logits, labels[batch_idx2])
            correct_num1 += cur_correct1
            total_num1 += cur_count1
            acc1 = correct_num1 * 100.0 / total_num1

            #train net2
            classifier2.train()
            optimizer_model2.zero_grad()
            output = classifier2(batch[batch_idx1,:], labels = labels[batch_idx1])
            logits = output.logits
            loss2 = computing_loss(logits, labels[batch_idx1])
            loss2.backward()
            optimizer_model2.step()
            loss2 = loss2 / batch_num
            loss2_value += loss2.item()

            cur_correct2, cur_count2 = compute_accuracy(logits, labels[batch_idx1])
            correct_num2 += cur_correct2
            total_num2 += cur_count2
            acc2 = correct_num2 * 100.0 / total_num2



        during_time = float(time.time() - start_time)
        print("net1:Step:%d, ACC:%.2f, Iter:%d, time:%.2f, loss:%.2f" \
                % (global_step, acc1, iter, during_time, loss1_value))
        train_acc_list1.append(acc1)
        train_loss_list1.append(loss1_value)

        print("net2:Step:%d, ACC:%.2f, Iter:%d, time:%.2f, loss:%.2f" \
                % (global_step, acc2, iter, during_time, loss2_value))
        train_acc_list2.append(acc2)
        train_loss_list2.append(loss2_value)

        global_step += 1

        tag_correct1, tag_total1, dev_tag_acc1, _, dev_f11 = \
            evaluate(dev_loader, classifier1, config.dev_file + '.' + str(global_step), dev_data)
        print("Dev: f1 = %d/%d = %.2f" % (tag_correct1, tag_total1, dev_f11))
        dev_list1.append(dev_f11)

        tag_correct2, tag_total2, dev_tag_acc2, _, dev_f12 = \
            evaluate(dev_loader, classifier2, config.dev_file + '.' + str(global_step), dev_data)
        print("Dev: f1 = %d/%d = %.2f" % (tag_correct2, tag_total2, dev_f12))
        dev_list2.append(dev_f12)

        tag_correct1, tag_total1, test_tag_acc1, _,test_f11 = \
            evaluate(test_loader, classifier1, config.test_file + '.' + str(global_step), test_data)
        print("Test: f1 = %d/%d = %.2f" % (tag_correct1, tag_total1, test_f11))
        test_list1.append(test_f11)

        tag_correct2, tag_total2, test_tag_acc2, _,test_f12 = \
            evaluate(test_loader, classifier2, config.test_file + '.' + str(global_step), test_data)
        print("Test: f1 = %d/%d = %.2f" % (tag_correct2, tag_total2, test_f12))
        test_list2.append(test_f12)
        

        if max(dev_f11, dev_f12) > best_f1:
            print("Exceed best acc: history = %.2f, current = %.2f" %(best_f1, max(dev_f11, dev_f12)))
            best_f1 = max(dev_f11, dev_f12)
            if best_f1 == dev_f11:
                best_val_model_test_f1 = test_f11
            else:
                best_val_model_test_f1 = test_f12
            
        print("Current best results: dev = %.2f,test = %.2f"%(best_f1, best_val_model_test_f1))    

def metaweightnet(data, meta_data, test_data, classifier, optimizer_model, noise_type, noise_ratio, config, model_name, times):
    
    class VNet(nn.Module):
        def __init__(self, input, hidden, output):
            super(VNet, self).__init__()
            self.linear1 = nn.Linear(input, hidden)
            self.relu1 = nn.ReLU(inplace=True)
            self.linear2 = nn.Linear(hidden, output)

        def forward(self, x):
            x = self.linear1(x)
            x = self.relu1(x)
            out = self.linear2(x)
            return torch.sigmoid(out)
    vnet = VNet(1, 100, 1).to("cuda")
    optimizer_vnet = torch.optim.Adam(vnet.parameters(), 1e-3, weight_decay=1e-4)

    train_loader = torch.utils.data.DataLoader(
        data, batch_size=config.train_batch_size, shuffle=True
    )
    meta_loader = torch.utils.data.DataLoader(
        meta_data, batch_size=config.train_batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=config.train_batch_size, shuffle=False
    )
    global_step = 0
    best_f1 = 0
    best_val_model_test_f1 = 0
    batch_num = int(np.ceil(len(data) / float(config.train_batch_size)))
    train_acc_list,train_loss_list, meta_list, meta_loss_list, test_list, test_loss_list = [], [], [], [], [], []

    meta_loader_iter = iter(meta_loader)
    for epoch in range(config.train_iters):
        start_time = time.time()
        print('Iteration: ' + str(epoch) + ', total batch num: ' + str(batch_num))
        batch_iter = 0

        correct_num, total_num, loss_value = 0.0, 0.0, 0.0
        classifier.train()
        for batch, labels in tqdm(train_loader):
            # bert_inputs, tags, masks = \
            #     batch_data_variable(onebatch, vocab)
            optimizer_vnet.zero_grad()
            with higher.innerloop_ctx(classifier, optimizer_model) as (meta_classifier, diffopt):
                if noise_type != 'none':
                    batch = torch.stack(batch, dim = 1)
                batch, labels = batch.to("cuda"), labels.to("cuda")
                outputs = meta_classifier(batch)
                cost = F.cross_entropy(outputs.logits, labels, reduction='none')
                cost_v = torch.reshape(cost,(len(cost), 1))
                v_lambda = vnet(cost_v.data)
                l_f_meta = torch.sum(cost * v_lambda) / len(cost_v)

                diffopt.step(l_f_meta)
                torch.cuda.empty_cache()
                try:
                    meta_batch = next(meta_loader_iter)
                except StopIteration:
                    meta_loader_iter = iter(meta_loader)
                    meta_batch = next(meta_loader_iter)
                outputs = meta_classifier(meta_batch[0].to('cuda'))
                meta_target = meta_batch[1].to('cuda')
                meta_loss = F.cross_entropy(outputs.logits, meta_target)
                meta_loss.backward()
            
            optimizer_vnet.step()

            optimizer_model.zero_grad()
            output = classifier(batch)
            cost = F.cross_entropy(output.logits, labels, reduction='none')
            cost_v = torch.reshape(cost,(len(cost), 1))
            with torch.no_grad():
                w_new = vnet(cost_v.data)

            loss = torch.sum(cost_v * w_new) / len(cost_v)
            loss.backward()
            optimizer_model.step()

            loss = loss / batch_num
            loss_value += loss.item()

            cur_correct, cur_count = compute_accuracy(output.logits, labels)
            correct_num += cur_correct
            total_num += cur_count
            acc = correct_num * 100.0 / total_num

            batch_iter += 1

        during_time = float(time.time() - start_time)
        print("Step:%d, ACC:%.2f, Iter:%d, batch:%d, time:%.2f, loss:%.2f" \
                % (global_step, acc, epoch, batch_iter-1, during_time, loss_value))
        train_acc_list.append(acc)
        train_loss_list.append(loss_value)

        global_step += 1

        tag_correct, tag_total, meta_tag_acc, meta_data_loss, meta_f1= \
            evaluate(meta_loader, classifier, config.dev_file + '.' + str(global_step), dev_data)
        print("Dev: acc = %d/%d = %.2f" % (tag_correct, tag_total, meta_f1))
        meta_list.append(meta_f1)
        meta_loss_list.append(meta_data_loss)

        tag_correct, tag_total, test_tag_acc, test_loss, test_f1 = \
            evaluate(test_loader, classifier, config.test_file + '.' + str(global_step), test_data)
        print("Test: acc = %d/%d = %.2f" % (tag_correct, tag_total, test_f1))
        test_list.append(test_f1)
        test_loss_list.append(test_loss)

        if meta_f1 > best_f1:
            print("Exceed best acc: history = %.2f, current = %.2f" %(best_f1, meta_f1))
            best_f1 = meta_f1
            best_val_model_test_f1 = test_f1
            # if config.save_after > 0 and iter > config.save_after:
            #     torch.save(classifier.state_dict(), config.save_model_path)
        print("Current best results: dev = %.2f,test = %.2f"%(best_f1, best_val_model_test_f1))

def joint_optimization(data, dev_data, test_data, classifier1, classifier2, optimizer_model1, optimizer_model2, noise_type, noise_ratio, config, computing_loss, model_name, times, priori_times):
    train_loader = torch.utils.data.DataLoader(
        data, batch_size=config.train_batch_size, shuffle=True
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_data, batch_size=config.train_batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=config.train_batch_size, shuffle=True
    )
    global_step = 0
    best_f1 = 0
    best_val_model_test_f1 = 0
    batch_num = int(np.ceil(len(data) / float(config.train_batch_size)))
    train_acc_list,train_loss_list, dev_list, dev_loss_list, test_list, test_loss_list = [], [], [], [], [], []

    relabel_epoch = 30
    soft_labels = []
    prob_list = []
    for i in range(len(data)):
        soft_labels.append(data[i][2])
        prob_list.append([])
    #first step
    for iter in range(relabel_epoch):
        start_time = time.time()
        print('Iteration: ' + str(iter) + ', total batch num: ' + str(batch_num))
        batch_iter = 0

        correct_num, total_num, loss_value = 0.0, 0.0, 0.0
        classifier1.train()
        for index, batch, labels in tqdm(train_loader):
            # bert_inputs, tags, masks = \
            #     batch_data_variable(onebatch, vocab)
            if noise_type != 'none':
                batch = torch.stack(batch, dim = 1)
            batch, labels = batch.to("cuda"), labels.to("cuda")
            soft_label = []
            for i in range(len(index)):
                soft_label.append(soft_labels[index[i]])
            soft_label = torch.tensor(soft_label ,dtype = torch.int64) 
            soft_label = soft_label.to('cuda')
            optimizer_model1.zero_grad()
            output = classifier1(batch, labels=soft_label)
            logits = output.logits
            probs = F.softmax(logits, dim=1)
            probs = probs.cpu().detach().numpy().tolist()
            loss = computing_loss(logits, soft_label)
            loss.backward()
            optimizer_model1.step()
            loss = loss / batch_num
            loss_value += loss.item()

            cur_correct, cur_count = compute_accuracy(logits, labels)
            correct_num += cur_correct
            total_num += cur_count
            acc = correct_num * 100.0 / total_num

            for i in range(len(index)):
                prob_list[index[i]].append(probs[i])

            batch_iter += 1

        during_time = float(time.time() - start_time)
        print("Step:%d, ACC:%.2f, Iter:%d, batch:%d, time:%.2f, loss:%.2f" \
                % (global_step, acc, iter, batch_iter-1, during_time, loss_value))
        train_acc_list.append(acc)
        train_loss_list.append(loss_value)
        loss_value = 0

        if iter % priori_times == (priori_times - 1) or iter == relabel_epoch - 1:
            prob_list= torch.tensor(prob_list,dtype=torch.float32)
            prob_soft_label = prob_list.mean(axis=1)
            pred_soft_labels = np.argmax(prob_soft_label, axis=1)
            pred_soft_labels = pred_soft_labels.cpu().detach().numpy().tolist()
            for i in range(len(soft_labels)):
                soft_labels[i] = pred_soft_labels[i]
            prob_list = []
            for i in range(len(data)):
                prob_list.append([])

    #second step
    for iter in range(config.train_iters):
        start_time = time.time()
        print('Iteration: ' + str(iter) + ', total batch num: ' + str(batch_num))
        batch_iter = 0

        correct_num, total_num, loss_value = 0.0, 0.0, 0.0
        classifier2.train()
        for index, batch, labels in tqdm(train_loader):
            # bert_inputs, tags, masks = \
            #     batch_data_variable(onebatch, vocab)
            if noise_type != 'none':
                batch = torch.stack(batch, dim = 1)
            batch, labels = batch.to("cuda"), labels.to("cuda")
            soft_label = []
            for i in range(len(index)):
                soft_label.append(soft_labels[index[i]])
            soft_label = torch.tensor(soft_label ,dtype = torch.int64) 
            soft_label = soft_label.to('cuda')
            optimizer_model2.zero_grad()
            output = classifier2(batch, labels=soft_label)
            logits = output.logits
            loss = F.cross_entropy(logits, soft_label)
            loss.backward()
            optimizer_model2.step()
            loss = loss / batch_num
            loss_value += loss.item()

            cur_correct, cur_count = compute_accuracy(logits, soft_label)
            correct_num += cur_correct
            total_num += cur_count
            acc = correct_num * 100.0 / total_num

            batch_iter += 1

        during_time = float(time.time() - start_time)
        print("Step:%d, ACC:%.2f, Iter:%d, batch:%d, time:%.2f, loss:%.2f" \
                % (global_step, acc, iter, batch_iter-1, during_time, loss_value))
        train_acc_list.append(acc)
        train_loss_list.append(loss_value)
        

        global_step += 1

        tag_correct, tag_total, dev_tag_acc, dev_loss, dev_f1= \
            evaluate(dev_loader, classifier2, config.dev_file + '.' + str(global_step), dev_data)
        print("Dev: f1 = %d/%d = %.2f" % (tag_correct, tag_total, dev_f1))
        dev_list.append(dev_f1)
        dev_loss_list.append(dev_loss)

        tag_correct, tag_total, test_tag_acc, test_loss, test_f1 = \
            evaluate(test_loader, classifier2, config.test_file + '.' + str(global_step), test_data)
        print("Test: f1 = %d/%d = %.2f" % (tag_correct, tag_total, test_f1))
        test_list.append(test_f1)
        test_loss_list.append(test_loss)

        if dev_f1 > best_f1:
            print("Exceed best f1: history = %.2f, current = %.2f" %(best_f1, dev_f1))
            best_f1 = dev_f1
            best_val_model_test_f1 = test_f1
            if config.save_after > 0 and iter > config.save_after:
                torch.save(classifier2.state_dict(), config.save_model_path)
        print("Current best results: dev = %.2f,test = %.2f"%(best_f1, best_val_model_test_f1))

def relabel(data, dev_data, test_data, classifier1, classifier2, optimizer_model1, optimizer_model2, noise_type, noise_ratio, config, computing_loss, model_name, times, priori_times):
    train_loader = torch.utils.data.DataLoader(
        data, batch_size=config.train_batch_size, shuffle=True
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_data, batch_size=config.train_batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=config.train_batch_size, shuffle=True
    )
    global_step = 0
    best_f1 = 0
    best_val_model_test_f1 = 0
    batch_num = int(np.ceil(len(data) / float(config.train_batch_size)))
    train_acc_list,train_loss_list, dev_list, dev_loss_list, test_list, test_loss_list = [], [], [], [], [], []

    # times = 0
    # priori_times = 8

    # if noise_ratio <0.33:
    #     change_rate = 1/(1 + np.exp(((-noise_ratio+0.33)**(1/2))*4.5))
    # else:
    #     change_rate = 1/(1 + np.exp((-(noise_ratio-0.33)**(1/2))*4.5))
    
    relabel_epoch = 30
    soft_labels = []
    prob_list = []
    for i in range(len(data)):
        soft_labels.append(data[i][2])
        prob_list.append([])
    #first step
    for iter in range(relabel_epoch):
        start_time = time.time()
        print('Iteration: ' + str(iter) + ', total batch num: ' + str(batch_num))
        batch_iter = 0

        correct_num, total_num, loss_value = 0.0, 0.0, 0.0
        classifier1.train()
        for index, batch, labels in tqdm(train_loader):
            # bert_inputs, tags, masks = \
            #     batch_data_variable(onebatch, vocab)
            if noise_type != 'none':
                batch = torch.stack(batch, dim = 1)
            batch, labels = batch.to("cuda"), labels.to("cuda")
            soft_label = []
            for i in range(len(index)):
                soft_label.append(soft_labels[index[i]])
            soft_label = torch.tensor(soft_label ,dtype = torch.int64) 
            soft_label = soft_label.to('cuda')
            optimizer_model1.zero_grad()
            output = classifier1(batch, labels=soft_label)
            logits = output.logits
            probs = F.softmax(logits, dim=1)
            probs = probs.cpu().detach().numpy().tolist()
            loss = computing_loss(logits, soft_label)
            loss.backward()
            optimizer_model1.step()
            loss = loss / batch_num
            loss_value += loss.item()

            cur_correct, cur_count = compute_accuracy(logits, labels)
            correct_num += cur_correct
            total_num += cur_count
            acc = correct_num * 100.0 / total_num

            for i in range(len(index)):
                prob_list[index[i]].append(probs[i])

            batch_iter += 1

        during_time = float(time.time() - start_time)
        print("Step:%d, ACC:%.2f, Iter:%d, batch:%d, time:%.2f, loss:%.2f" \
                % (global_step, acc, iter, batch_iter-1, during_time, loss_value))
        train_acc_list.append(acc)
        train_loss_list.append(loss_value)
        loss_value = 0


        if iter % priori_times == (priori_times - 1) and iter != relabel_epoch - 1:
            prob_list= torch.tensor(prob_list,dtype=torch.float32)
            prob_soft_label = prob_list.mean(axis=1)
            pred_soft_labels = np.argmax(prob_soft_label, axis=1)
            pred_soft_labels = pred_soft_labels.cpu().detach().numpy().tolist()
            for i in range(len(soft_labels)):
                soft_labels[i] = pred_soft_labels[i]
            prob_list = []
            for i in range(len(data)):
                prob_list.append([])

        if iter == relabel_epoch - 1:
            change_num = noise_ratio * len(soft_labels)
            change_idx, unchange_idx = [], []
            change_idx_pre, unchange_idx_pre = [], []
            unchange_idx_pro = []
            for i in range(len(soft_labels)):
                if soft_labels[i] != data[i][2]:
                    change_idx.append(i)
                    change_idx_pre.append(prob_soft_label[i][soft_labels[i]])
                else:
                    unchange_idx.append(i)
                    unchange_idx_pro.append(prob_soft_label[i])
                    unchange_idx_pre.append(prob_soft_label[i][soft_labels[i]])
            change_rank = np.argsort(change_idx_pre)
            unchange_rank = np.argsort(unchange_idx_pre)
            return_num = len(change_idx) - change_num
            if return_num > 0:
                return_idx = change_rank[:int(return_num)]
                for i in range(len(return_idx)):
                    soft_labels[change_idx[return_idx[i]]] = data[change_idx[return_idx[i]]][2]
            if return_num <=0:
                extra_num = - return_num
                extra_idx = unchange_rank[:int(extra_num)]
                for i in range(len(extra_idx)):
                    unchange_idx_pro[extra_idx[i]][soft_labels[unchange_idx[extra_idx[i]]]] = 0
                    soft_labels[unchange_idx[extra_idx[i]]] = np.argmax(unchange_idx_pro[extra_idx[i]]).cpu().detach().numpy().tolist()



    #second step
    for iter in range(config.train_iters):
        start_time = time.time()
        print('Iteration: ' + str(iter) + ', total batch num: ' + str(batch_num))
        batch_iter = 0

        correct_num, total_num, loss_value = 0.0, 0.0, 0.0
        classifier2.train()
        for index, batch, labels in tqdm(train_loader):
            # bert_inputs, tags, masks = \
            #     batch_data_variable(onebatch, vocab)
            if noise_type != 'none':
                batch = torch.stack(batch, dim = 1)
            batch, labels = batch.to("cuda"), labels.to("cuda")
            soft_label = []
            for i in range(len(index)):
                soft_label.append(soft_labels[index[i]])
            soft_label = torch.tensor(soft_label ,dtype = torch.int64) 
            soft_label = soft_label.to('cuda')
            optimizer_model2.zero_grad()
            output = classifier2(batch, labels=soft_label)
            logits = output.logits
            loss = F.cross_entropy(logits, soft_label)
            loss.backward()
            optimizer_model2.step()
            loss = loss / batch_num
            loss_value += loss.item()

            cur_correct, cur_count = compute_accuracy(logits, soft_label)
            correct_num += cur_correct
            total_num += cur_count
            acc = correct_num * 100.0 / total_num

            batch_iter += 1

        during_time = float(time.time() - start_time)
        print("Step:%d, ACC:%.2f, Iter:%d, batch:%d, time:%.2f, loss:%.2f" \
                % (global_step, acc, iter, batch_iter-1, during_time, loss_value))
        train_acc_list.append(acc)
        train_loss_list.append(loss_value)
        

        global_step += 1

        tag_correct, tag_total, dev_tag_acc, dev_loss, dev_f1= \
            evaluate(dev_loader, classifier2, config.dev_file + '.' + str(global_step), dev_data)
        print("Dev: f1 = %d/%d = %.2f" % (tag_correct, tag_total, dev_f1))
        dev_list.append(dev_f1)
        dev_loss_list.append(dev_loss)

        tag_correct, tag_total, test_tag_acc, test_loss, test_f1 = \
            evaluate(test_loader, classifier2, config.test_file + '.' + str(global_step), test_data)
        print("Test: f1 = %d/%d = %.2f" % (tag_correct, tag_total, test_f1))
        test_list.append(test_f1)
        test_loss_list.append(test_loss)

        if dev_f1 > best_f1:
            print("Exceed best f1: history = %.2f, current = %.2f" %(best_f1, dev_f1))
            best_f1 = dev_f1
            best_val_model_test_f1 = test_f1
            if config.save_after > 0 and iter > config.save_after:
                torch.save(classifier2.state_dict(), config.save_model_path)
        print("Current best results: dev = %.2f,test = %.2f"%(best_f1, best_val_model_test_f1))

def evaluate(data, classifier, outputFile, raw_data):
    start = time.time()
    classifier.eval()
    batch_num = int(np.ceil(len(raw_data) / float(config.train_batch_size)))
    # output = open(outputFile, 'w', encoding='utf-8')
    # tag_correct, tag_total = 0, 0
    total_num = 0.0
    correct_num = 0.0
    eva_loss = 0.0
    y_predicts = []
    y_labels = []

    with torch.no_grad():
        for batch, labels in tqdm(data):
            batch, labels = batch.to("cuda"), labels.to("cuda")
            pred_tags = classifier(batch)[0]
            output = classifier(batch, labels=labels)
            logits = output.logits
            val_loss = F.cross_entropy(logits, labels)
            val_loss = val_loss / batch_num
            eva_loss += val_loss.item()

            predict = torch.argmax(logits, axis=1)
            for i in range(len(predict)):
                y_predicts.append(predict[i].cpu().detach().numpy().tolist())
            for i in range(len(labels)):
                y_labels.append(labels[i].cpu().detach().numpy().tolist())
            cur_correct, cur_count = compute_accuracy(pred_tags, labels)
            correct_num += cur_correct
            total_num += cur_count
            # acc = correct_num * 100.0 / total_num

    # output.close()

    acc = correct_num * 100.0 / total_num
    y_macro_F1 = metrics.f1_score(y_labels, y_predicts, labels=[0,1,2,3,4],average='macro')
    end = time.time()
    during_time = float(end - start)
    print("sentence num: %d,  classifier time = %.2f " % (len(data), during_time))

    return correct_num, total_num, acc, eva_loss, y_macro_F1

class Optimizer:
    def __init__(self, parameter, config):
        self.optim = torch.optim.Adam(parameter, lr=config.learning_rate, betas=(config.beta_1, config.beta_2),
                                      eps=config.epsilon)
        decay, decay_step = config.decay, config.decay_steps
        l = lambda epoch: decay ** (epoch // decay_step)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=l)

    def step(self):
        self.optim.step()
        self.schedule()
        self.optim.zero_grad()

    def schedule(self):
        self.scheduler.step()

    def zero_grad(self):
        self.optim.zero_grad()

    @property
    def lr(self):
        return self.scheduler.get_lr()

def main(noisy_data, dev_set, test_set, model_name, noise_type, noise_ratio, meta_num, config , times):

    config_class, model_class = BertConfig, BertForSequenceClassification
    bert_config = config_class.from_pretrained("./my-bert", num_labels=5)
    classifier = model_class.from_pretrained("./my-bert", config=bert_config)
    classifier.cuda()
    optimizer_model = torch.optim.Adam(classifier.parameters(), lr=1e-5)

    
    start_train = time.time()

    if model_name == 'cross_entropy':
        def computing_loss(logtis, labels):
            loss = F.cross_entropy(logtis, labels)
            return loss
        cross_entropy(noisy_data, dev_set, test_set, classifier, optimizer_model, noise_type, noise_ratio, config, computing_loss, model_name, times)

    elif model_name == 'symmetric_crossentropy':
        def computing_loss(logits, labels):
            alpha, beta = 1, 0.1
            y_true_1 = F.one_hot(labels, NUM_CLASSES[dataset])
            y_pred_1 = F.softmax(logits, dim=1)

            y_true_2 = F.one_hot(labels, NUM_CLASSES[dataset])
            y_pred_2 = F.softmax(logits, dim=1)

            y_pred_1 = torch.clamp(y_pred_1, 1e-7, 1.0)
            y_true_2 = torch.clamp(y_true_2, 1e-4, 1.0)
            loss1 = -torch.sum(y_true_1 * F.log_softmax(y_pred_1, dim=1), axis=1)
            loss2 = -torch.sum(y_pred_2 * F.log_softmax(y_true_2.type(torch.float), dim=1), axis=1)
            loss = alpha*loss1 + beta*loss2
            loss = loss.sum()/loss.shape[0]
            return loss
        cross_entropy(noisy_data, dev_set, test_set, classifier, optimizer_model, noise_type, noise_ratio, config, computing_loss, model_name, times)

    elif model_name == 'generalized_crossentropy':
        def computing_loss(logits, labels):    
            q = 0.7
            ytrue_tmp = F.one_hot(labels, NUM_CLASSES[dataset])
            ypred_tmp = F.softmax(logits, dim=1)
            loss = (1 - torch.pow(torch.sum(ytrue_tmp*ypred_tmp, axis=1), q)) / q
            loss = loss.sum()/loss.shape[0]
            return loss
        cross_entropy(noisy_data, dev_set, test_set, classifier, optimizer_model, noise_type, noise_ratio, config, computing_loss, model_name, times)

    elif model_name == 'bootstrap_soft':
        def computing_loss(logits, labels):
            beta = 0.9
            y_pred_softmax = F.softmax(logits, dim=1)
            y_true_onehot = F.one_hot(labels, NUM_CLASSES[dataset])
            y_true_modified = beta * y_true_onehot + (1. - beta) * y_pred_softmax
            loss = -torch.sum(y_true_modified * F.log_softmax(logits, dim=1),axis=1)
            loss = loss.sum()/loss.shape[0]
            return loss
        cross_entropy(noisy_data, dev_set, test_set, classifier, optimizer_model, noise_type, noise_ratio, config, computing_loss, model_name, times)
    
    elif model_name == 'joint_optimization':
        def computing_loss(logits, labels):
            probs = F.softmax(logits, dim=1)
            avg_probs = torch.mean(probs, axis=0)
            y_true_onehot = F.one_hot(labels, NUM_CLASSES[dataset])

            l_c = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * y_true_onehot, dim=1))
            l_p = -torch.sum(torch.log(avg_probs) * p)
            l_e = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * probs, dim=1))

            per_example_loss = l_c + 1.2 * l_p + 0.8 * l_e
            return per_example_loss 

        p = np.ones(NUM_CLASSES[dataset], dtype=np.float32) / float(NUM_CLASSES[dataset])
        p = torch.tensor(p, dtype=torch.float).cuda()

        #net1
        classifier1 = model_class.from_pretrained("./my-bert", config=bert_config)
        classifier1.cuda()
        optimizer_model1 = torch.optim.Adam(classifier1.parameters(), lr=1e-5)
        #net2
        classifier2 = model_class.from_pretrained("./my-bert", config=bert_config)
        classifier2.cuda()
        optimizer_model2 = torch.optim.Adam(classifier2.parameters(), lr=1e-5)

        noisy_data1 = [
            (i, noisy_data[i][0], int(noisy_data[i][1]))
            for i in range(len(noisy_data))
        ]  

        joint_optimization(noisy_data1, dev_set, test_set, classifier1, classifier2, optimizer_model1, optimizer_model2, noise_type, noise_ratio, config, computing_loss, model_name, times, priori_times = 8)
        
    elif model_name == 'coteaching':
        def computing_loss(logtis, labels):
            loss = F.cross_entropy(logtis, labels)
            return loss        
        
        #net1
        classifier1 = model_class.from_pretrained("./my-bert", config=bert_config)
        classifier1.cuda()
        optimizer_model1 = torch.optim.Adam(classifier1.parameters(), lr=1e-5)
        #net2
        classifier2 = model_class.from_pretrained("./my-bert", config=bert_config)
        classifier2.cuda()
        optimizer_model2 = torch.optim.Adam(classifier2.parameters(), lr=1e-5)

        coteaching(noisy_data, dev_set, test_set, classifier1, classifier2, optimizer_model1, optimizer_model2, noise_type, noise_ratio, config, computing_loss, model_name, times)
    
    elif model_name == 'mwnet':
        data_list_val = {}
        for j in range(NUM_CLASSES[dataset]):
            data_list_val[j] = [i for i, sentence in enumerate(dev_set) if sentence[1] == j]
        meta_num_list = [int(meta_num / NUM_CLASSES[dataset])] * NUM_CLASSES[dataset] 
        idx_to_meta = []
        for cls_idx, sentence_id_list in data_list_val.items():
            np.random.shuffle(sentence_id_list)
            sentence_num = meta_num_list[int(cls_idx)]
            idx_to_meta.extend(sentence_id_list[:sentence_num])
        meta_set = [
            (dev_set[idx_to_meta[i]][0], dev_set[idx_to_meta[i]][1])
            for i in range(len(idx_to_meta))
        ]    

        metaweightnet(noisy_data, meta_set, test_set, classifier, optimizer_model, noise_type, noise_ratio, config, model_name, times)

    elif model_name == 'relabel':
        def computing_loss(logits, labels):
            probs = F.softmax(logits, dim=1)
            avg_probs = torch.mean(probs, axis=0)
            y_true_onehot = F.one_hot(labels, NUM_CLASSES[dataset])

            l_c = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * y_true_onehot, dim=1))
            l_p = -torch.sum(torch.log(avg_probs) * p)
            l_e = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * probs, dim=1))

            per_example_loss = l_c + 1.2 * l_p + 0.8 * l_e
            return per_example_loss 

        p = np.ones(NUM_CLASSES[dataset], dtype=np.float32) / float(NUM_CLASSES[dataset])
        p = torch.tensor(p, dtype=torch.float).cuda()
        #net1
        classifier1 = model_class.from_pretrained("./my-bert", config=bert_config)
        classifier1.cuda()
        optimizer_model1 = torch.optim.Adam(classifier1.parameters(), lr=1e-5)
        #net2
        classifier2 = model_class.from_pretrained("./my-bert", config=bert_config)
        classifier2.cuda()
        optimizer_model2 = torch.optim.Adam(classifier2.parameters(), lr=1e-5)

        noisy_data1 = [
            (i, noisy_data[i][0], int(noisy_data[i][1]))
            for i in range(len(noisy_data))
        ]  

        relabel(noisy_data1, dev_set, test_set, classifier1, classifier2, optimizer_model1, optimizer_model2, noise_type, noise_ratio, config, computing_loss, model_name, times, priori_times = 8)

    print('Totail training duration: {:3.2f}h'.format((time.time()-start_train)/3600))

if __name__ == '__main__':
    torch.manual_seed(888)
    torch.cuda.manual_seed(888)
    random.seed(888)
    np.random.seed(888)

    # gpu
    gpu = torch.cuda.is_available()
    print(torch.__version__)
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('-d', '--dataset', required=False, type=str, default='sst5',
    help="""dataset name: 'sst5', 
                        'ar', 
                        'yr', 
                        'DBpedia',""")
    argparser.add_argument('-m', '--model_name', required=False, type=str, default='relabel',
    help="""Model name: 'cross_entropy', 
                        'symmetric_crossentropy', 
                        'generalized_crossentropy', 
                        'bootstrap_soft',  
                        'joint_optimization', 
                        'coteaching',
                        'mwnet',
                        'relabel'""")
                        
    argparser.add_argument('-n', '--noise_type', required=False, type=str, default='uniform',
        help="""Noise type: 'none',
                        'uniform',
                        'locally-concentrated'
                        'label-dependent',
                        'instance-dependent(r)',
                        'instance-dependent(l)',
                        'instance-dependent(g)',""")
    argparser.add_argument('-r', '--noise_ratio', required=False, type=float, default=50,
        help=" noise ratio in percentage between 0-100")
    argparser.add_argument('-num', '--meta_num', required=False, type=int, default=200,
        help=" meta number in percentage between 0-1100, best is a mutiple of 5")
    argparser.add_argument('--config_file', default='default.cfg')
    argparser.add_argument('--thread', default=1, type=int, help='thread num')
    argparser.add_argument('--gpu', default=0, type=int, help='Use id of gpu, -1 if cpu.')

    args, extra_args = argparser.parse_known_args()
    torch.set_num_threads(args.thread)
    config = Configurable(args.config_file, extra_args)
    dataset = args.dataset
    model_name = args.model_name
    noise_type = args.noise_type
    noise_ratio = args.noise_ratio / 100
    NUM_CLASSES = {'sst5':5, 'ar':5, 'yr':5, 'DBpedia':14}
    meta_num = args.meta_num
    

    config.use_cuda = False
    gpu_id = -1
    if gpu and args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        config.use_cuda = True
        print("GPU ID: ", args.gpu)
        gpu_id = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES']='0'

    tokenizer_class = BertTokenizer
    tokenizer = tokenizer_class.from_pretrained("./my-bert")

    data = read_corpus(config.train_file)
    dev_data = read_corpus(config.dev_file)
    test_data = read_corpus(config.test_file)
    train_set = SentimentDataset(data)
    dev_set = SentimentDataset(dev_data)
    test_set = SentimentDataset(test_data)

    noisy_data = get_noisy_data(train_set, noise_type, noise_ratio, dataset)

    main(noisy_data, dev_set, test_set, model_name, noise_type, noise_ratio, meta_num, config, times)

