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
from sklearn.model_selection import KFold
from transformers import AdamW, get_linear_schedule_with_warmup, BertConfig, BertForSequenceClassification, BertTokenizer


def rpad(array, n=70):
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
    def __init__(self, data):

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

def train(data, dev_data, test_data, classifier, optimizer_model, config, dev_index, train_pre):
    train_loader = torch.utils.data.DataLoader(
        data, batch_size=config.train_batch_size, shuffle=True
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_data, batch_size=2000, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=config.train_batch_size, shuffle=True
    )
    global_step = 0
    best_acc = 0
    best_val_model_test_acc = 0
    batch_num = int(np.ceil(len(data) / float(config.train_batch_size)))

    
    for iter in range(config.train_iters):
        start_time = time.time()
        print('Iteration: ' + str(iter) + ', total batch num: ' + str(batch_num))
        batch_iter = 0

        correct_num, total_num, loss_value = 0.0, 0.0, 0.0
        classifier.train()
        for batch, labels in tqdm(train_loader):
            # bert_inputs, tags, masks = \
            #     batch_data_variable(onebatch, vocab)
            # batch = torch.stack(batch, dim = 1)
            batch, labels = batch.to("cuda"), labels.to("cuda")
            optimizer_model.zero_grad()
            output = classifier(batch, labels=labels)
            loss = output.loss
            logits = output.logits
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

        loss_value = 0

        global_step += 1

        tag_correct, tag_total, dev_tag_acc = \
            evaluate_dev(dev_loader, classifier, dev_index, train_pre)
        print("Dev: acc = %d/%d = %.2f" % (tag_correct, tag_total, dev_tag_acc))

        tag_correct, tag_total, test_tag_acc = \
            evaluate(test_loader, classifier)
        print("Test: acc = %d/%d = %.2f" % (tag_correct, tag_total, test_tag_acc))
        if dev_tag_acc > best_acc:
            print("Exceed best acc: history = %.2f, current = %.2f" %(best_acc, dev_tag_acc))
            best_acc = dev_tag_acc
            best_val_model_test_acc = test_tag_acc
            if config.save_after > 0 and iter > config.save_after:
                torch.save(classifier.state_dict(), config.save_model_path)
        print("Current best results: dev = %.2f,test = %.2f"%(best_acc, best_val_model_test_acc))

def evaluate(data, classifier):
    start = time.time()
    classifier.eval()
    # output = open(outputFile, 'w', encoding='utf-8')
    # tag_correct, tag_total = 0, 0
    total_num = 0.0
    correct_num = 0.0

    with torch.no_grad():
        for batch, labels in tqdm(data):
            batch, labels = batch.to("cuda"), labels.to("cuda")
            pred_tags = classifier(batch)[0]
            # for inst, bmatch in batch_variable_inst(onebatch, pred_tags, vocab):
            #     printInstance(output, inst)
            #     tag_total += 1
            #     if bmatch: tag_correct += 1
            #     count += 1

            cur_correct, cur_count = compute_accuracy(pred_tags, labels)
            correct_num += cur_correct
            total_num += cur_count
            # acc = correct_num * 100.0 / total_num

    # output.close()

    acc = correct_num * 100.0 / total_num

    end = time.time()
    during_time = float(end - start)
    print("sentence num: %d,  classifier time = %.2f " % (len(data), during_time))

    return correct_num, total_num, acc

def evaluate_dev(data, classifier, dev_index, train_pre):
    start = time.time()
    classifier.eval()
    # output = open(outputFile, 'w', encoding='utf-8')
    # tag_correct, tag_total = 0, 0
    total_num = 0.0
    correct_num = 0.0

    with torch.no_grad():
        for batch, labels in tqdm(data):
            batch, labels = batch.to("cuda"), labels.to("cuda")
            pred_tags = classifier(batch)[0]
            # for inst, bmatch in batch_variable_inst(onebatch, pred_tags, vocab):
            #     printInstance(output, inst)
            #     tag_total += 1
            #     if bmatch: tag_correct += 1
            #     count += 1

            cur_correct, cur_count = compute_accuracy(pred_tags, labels)
            correct_num += cur_correct
            total_num += cur_count
            m = torch.nn.Softmax(dim = 1)
            pred_pro = (m(pred_tags)).cpu().numpy()
            for i in range(len(dev_index)):
                train_pre[dev_index[i]] = pred_pro[i].tolist()

    # output.close()

    acc = correct_num * 100.0 / total_num

    end = time.time()
    during_time = float(end - start)
    print("sentence num: %d,  classifier time = %.2f " % (len(data), during_time))

    return correct_num, total_num, acc

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
    argparser.add_argument('-n', '--noise_type', required=False, type=str, default='uniform',
        help="""Noise type: 'none',
                        'random',
                        'uniform',
                        'locally-concentrated'
                        'class-dependent',
                        'feature-dependent',
                        'probability_rank'""")
    argparser.add_argument('-r', '--noise_ratio', required=False, type=int, default=20,
        help=" noise ratio in percentage between 0-100")
    argparser.add_argument('--config_file', default='default.cfg')
    argparser.add_argument('--thread', default=1, type=int, help='thread num')
    argparser.add_argument('--gpu', default=-1, type=int, help='Use id of gpu, -1 if cpu.')

    args, extra_args = argparser.parse_known_args()
    torch.set_num_threads(args.thread)
    config = Configurable(args.config_file, extra_args)

    noise_type = args.noise_type
    noise_ratio = args.noise_ratio / 100


    config.use_cuda = False
    gpu_id = -1
    if gpu and args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        config.use_cuda = True
        print("GPU ID: ", args.gpu)
        gpu_id = args.gpu



    data = read_corpus(config.train_file)
    test_data = read_corpus(config.test_file)

    #5-fold
    split_num = 5
    times = 0
    train_pre = [None] * len(data)
    kf = KFold(n_splits = split_num, shuffle = True, random_state = 0)
    split_d = kf.split(data)
    for train_index, dev_index in split_d:
        train_dataset, dev_dataset = [], []
        times += 1      
        config_class, model_class, tokenizer_class = BertConfig, BertForSequenceClassification, BertTokenizer
        bert_config = config_class.from_pretrained("./my-bert", num_labels=5)
        tokenizer = tokenizer_class.from_pretrained("./my-bert")
        classifier = model_class.from_pretrained("./my-bert", config=bert_config)
        classifier.cuda()
        optimizer_model = torch.optim.Adam(classifier.parameters(), lr=1e-5)
        for a in range(len(train_index)):
            train_dataset.append(data[train_index[a]])
        for b in range(len(dev_index)):
            dev_dataset.append(data[dev_index[b]])

        train_data = train_dataset
        dev_data = dev_dataset

        train_set = SentimentDataset(train_data)
        dev_set = SentimentDataset(dev_data)
        test_set = SentimentDataset(test_data)


        train(train_set, dev_set, test_set, classifier, optimizer_model, config, dev_index, train_pre)
    filename='sst5_bert_pre.json'
    with open(filename,'w') as file_obj:
        json.dump(train_pre,file_obj)
