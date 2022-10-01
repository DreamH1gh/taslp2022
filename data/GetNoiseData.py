from dataclasses import replace
import numpy as np
import random
import copy
import json
from data.Dataloader import *
from sklearn.cluster import KMeans
from numpy.testing import assert_array_almost_equal


def get_noisy_data(data, noise_type, noise_ratio):
    if noise_ratio > 100:
        noise_ratio = 100
    elif noise_ratio < 0 :
        noise_ratio = 0
    num_classes = 5  # or 14

    if noise_ratio == 0 or noise_type == 'none':
        data_noisy = data
    
    elif noise_type == 'random':
        noise_pre = []
        for i in range(len(data)):
            a = (noise_ratio / (num_classes - 1)) * np.ones(num_classes)
            a[data.data[i][1]] = 1 - noise_ratio
            noise_pre.append(a.tolist())


        change_num = int(len(data) * noise_ratio)
        change_idx = np.random.choice(len(data), change_num, replace = False)
                
        data_noisy = make_FDorPR_noise(data, noise_pre, change_idx)

    elif noise_type == 'uniform':
        P = noise_ratio / (num_classes - 1) * np.ones((num_classes, num_classes))
        np.fill_diagonal(P, (1 - noise_ratio) * np.ones(num_classes))
        data_noisy = make_noise(data, P)

    elif noise_type == 'locally-concentrated':
        if num_classes == 2:
            P = noise_ratio / (num_classes - 1) * np.ones((num_classes, num_classes))
            np.fill_diagonal(P, (1 - noise_ratio) * np.ones(num_classes))            
        if num_classes >= 2:
            P = np.zeros((num_classes, num_classes)) 
            for i in range(num_classes):
                if i == 0:
                    P[i][i] = 1 - noise_ratio
                    P[i][i + 1] = 1 - P[i][i]
                elif i == (num_classes - 1):
                    P[i][i] = 1 - noise_ratio
                    P[i][i - 1] = 1 - P[i][i]
                else:
                    P[i][i] = 1 - noise_ratio
                    P[i][i + 1] = (1 - P[i][i]) / 2
                    P[i][i - 1] = (1 - P[i][i]) / 2
        data_noisy = make_noise(data, P)

    elif noise_type == 'class-dependent':

        filename = 'sst5_bert_pre.json'
        with open(filename) as file_obj:
            noise_pre = json.load(file_obj)
        tags = []
        for i in range(len(data.data)):
            tags.append(data.data[i][1])
        max_pre = []
        for i in range(len(noise_pre)):
            max_pre.append(noise_pre[i].index(max(noise_pre[i])))

        P = np.zeros((num_classes, num_classes)) 
        P_class = np.zeros((num_classes, num_classes)) 
        for i in range(len(tags)):
            P[tags[i]][max_pre[i]] += 1 
        label_num = Counter(tags)
        for i in range(len(label_num)):
            P[i] = P[i] / label_num[i]
        for i in range(num_classes):
            for j in range(num_classes):
                if i != j:
                    P_class[i][j] = (P[i][j] / (1 - P[i][i])) * noise_ratio
            P_class[i][i] = 1 - noise_ratio
        data_noisy = make_noise(data, P_class)

    elif noise_type == 'feature-dependent':

        filename = 'sst5_bert_pre.json'
        with open(filename) as file_obj:
            noise_pre = json.load(file_obj)
        change_num = int(len(data) * noise_ratio)
        change_idx = np.random.choice(len(data), change_num, replace = False)
                
        data_noisy = make_FDorPR_noise(data, noise_pre, change_idx)

    elif noise_type == 'feature_rank':

        filename = 'sst5_bert_pre.json'
        with open(filename) as file_obj:
            noise_pre = json.load(file_obj)
        noise_pre_class = [[], [], [], [], []]
        noise_pre_class_idx = [[], [], [], [], []]
        for i in range(len(data.data)):
            noise_pre_class[data.data[i][1]].append(noise_pre[i][data.data[i][1]])
            noise_pre_class_idx[data.data[i][1]].append(i)
        noise_pre_class_rank = [[], [], [], [], []]
        for i in range(len(noise_pre_class)):
            noise_pre_class_rank[i] = [noise_pre_class_idx[i][index] for index, value in sorted(list(enumerate(noise_pre_class[i])), key=lambda x:x[1])]
        change_idx = []
        for i in range(len(noise_pre_class_rank)):
            change_num = int(len(noise_pre_class_rank[i]) * noise_ratio)
            change_idx = change_idx + noise_pre_class_rank[i][:change_num]
                
        data_noisy = make_FDorPR_noise(data, noise_pre, change_idx)

    elif noise_type == 'probability_rank':

        filename = 'sst5_bert_pre.json'
        with open(filename) as file_obj:
            noise_pre = json.load(file_obj)
        probability_rank = get_rank(data, noise_pre)
        change_num = int(len(data) * noise_ratio)
        change_idx = probability_rank[:change_num]
        data_noisy = make_FDorPR_noise(data, noise_pre, change_idx)

    return data_noisy
    

def make_noise(data, P):
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    tags = []
    for i in range(len(data.data)):
        tags.append(data.data[i][1])
    tags_arr = np.array(tags)
    tags_noisy = copy.deepcopy(tags_arr)
    num_classes = P.shape[0]
    
    for i in range(num_classes):
        idx = np.where(tags_arr == i)
        idx = list(idx[0])
        n_samples = len(idx)
        for j in range(num_classes):
            if i != j:
                n_noisy = int(n_samples * P[i,j])
                noisy_idx = np.random.choice(len(idx), n_noisy, replace = False)
                for m in range(len(noisy_idx)):
                    tags_noisy[idx[noisy_idx[m]]] = j
                idx = np.delete(idx, noisy_idx)

    data_noisy = [
        (data.data[i][0], str(tags_noisy[i]))
        for i in range(len(data.data))
    ]
    
    return data_noisy

def get_rank(data, noise_pre):
    tags = []
    for i in range(len(data.data)):
        tags.append(data.data[i][1])
    tags_arr = np.array(tags)
    ids_arr = []
    for i in range(len(tags_arr)):
        ids_arr.append(tags_arr[i])
    right_pro = []
    for i in range(len(ids_arr)):
        right_pro.append(noise_pre[i][ids_arr[i]])
    probability_rank = [index for index, value in sorted(list(enumerate(right_pro)), key=lambda x:x[1])]
    
    
    return probability_rank

def make_FDorPR_noise(data, noise_pre, change_idx):
    tags = []
    for i in range(len(data.data)):
        tags.append(data.data[i][1])
    tags_arr = np.array(tags)
    tags_noisy = copy.deepcopy(tags_arr)
    data_noisy = copy.deepcopy(data)
    num_classes = len(noise_pre[0])
    index = []
    for i in range(num_classes):
        index.append(i)
    
    for i in range(len(change_idx)):
        change_id = change_idx[i]
        change_probabilities = noise_pre[change_id]
        true_id = tags_arr[change_id]
        for i in range(len(change_probabilities)):
            if i != true_id:
                change_probabilities[i] = change_probabilities[i] / (1 - change_probabilities[true_id])
        change_probabilities[true_id] = 0
        change2id = rand_pick(index, change_probabilities)
        tags_noisy[change_id] = change2id
    data_noisy = [
        (data.data[i][0], str(tags_noisy[i]))
        for i in range(len(data.data))
    ]
        

    return data_noisy

def rand_pick(index, probabilities):
    x = random.uniform(0,1)
    cumprob = 0.0
    for item, item_pro in zip(index, probabilities):
        cumprob += item_pro
        if x < cumprob:
            break
    return item
