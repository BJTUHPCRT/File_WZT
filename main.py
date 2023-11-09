# coding=utf-8
import matplotlib.pyplot as plt
import torch
import numpy as np

import random
import datetime
import time

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import os

import utils
from dataloader import Dataloader
from model import WGCN, RMF, CMF, MODEL, LRMF, ELF

seed = 2022
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现。


#

def get_model():
    if args.model == 'GCN':
        model = WGCN(args, dataloader).to(device=args.device)
        optimizer = torch.optim.Adam([
            {'params': model.embedding_item.parameters(), 'weight_decay': 1, 'lr': 0.01},
            {'params': model.embedding_user.parameters(), 'weight_decay': 1, 'lr': 0.01},
            {'params': model.features['user'].parameters(), 'weight_decay': 1, 'lr': 0.01},
            {'params': model.features['service'].parameters(), 'weight_decay': 1, 'lr': 0.01},

            {'params': model.transfrom.parameters(), 'weight_decay': 0.001, 'lr': 0.001},
            # {'params': model.transfrom2.parameters(), 'weight_decay': 0.001, 'lr': 0.001},
            {'params': model.attention.parameters(), 'weight_decay': 0.001, 'lr': 0.001},
            # {'params': model.rep_to_vector.parameters(), 'weight_decay': 0.001, 'lr': 0.001},
            {'params': model.wsdl_to_vector.parameters(), 'weight_decay': 0.001, 'lr': 0.001},
            # {'params': model.user_and_rep_to_user.parameters(), 'weight_decay': 0.001, 'lr': 0.001},

            # {'params': model.wsdl_to_vec.parameters(), 'weight_decay': 0.001, 'lr': 0.001},

            # {'params': model.embedding_item.parameters(), 'weight_decay': 0, 'lr': 0.01},
            # {'params': model.embedding_user.parameters(), 'weight_decay': 0, 'lr': 0.01},
            # {'params': model.features['user'].parameters(), 'weight_decay': 0, 'lr': 0.01},
            # {'params': model.features['service'].parameters(), 'weight_decay': 0, 'lr': 0.01},
            #
            # {'params': model.transfrom.parameters(), 'weight_decay': 0, 'lr': 0.001},
            # {'params': model.attention.parameters(), 'weight_decay': 0, 'lr': 0.001},
        ])
    else:
        if args.model == 'RMF':
            model = RMF(args, dataloader).to(device=args.device)
        elif args.model == 'LRMF':
            model = LRMF(args, dataloader).to(device=args.device)
        elif args.model == 'CMF':
            model = CMF(args, dataloader).to(device=args.device)
        elif args.model == 'PMF':
            model = MODEL(args, dataloader).to(device=args.device)
        elif args.model == 'ELF':
            model = ELF(args, dataloader).to(device=args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    print('Model Initialization Finish!!!')

    return model, optimizer


def test(model, model_name):
    model.eval()
    with torch.no_grad():
        users, items = model.getEmbedding()
        pre_qos = torch.mm(users, items.t())
        if model_name == 'ELF':
            pre_qos = 0.9 * torch.mm(users, items.t()) + \
                      0.1 * torch.sqrt(torch.sum((users[:, None] - items[None, :]) ** 2, dim=2))
        pre_qos = pre_qos.cpu().numpy()
        # pre_qos = utils.data_deprocess(pre_qos)
        qos_label = utils.data_deprocess(dataloader.qosdata)

        pre_qos_train = pre_qos[dataloader.train_mask == 1]
        pre_qos_test = pre_qos[dataloader.train_mask == 0]
        true_qos_train = qos_label[dataloader.train_mask == 1]
        true_qos_test = qos_label[dataloader.train_mask == 0]

        train_rmse = mean_squared_error(pre_qos_train, true_qos_train) ** 0.5
        train_mae = mean_absolute_error(pre_qos_train, true_qos_train)
        test_rmse = mean_squared_error(pre_qos_test, true_qos_test) ** 0.5
        test_mae = mean_absolute_error(pre_qos_test, true_qos_test)
        return round(train_rmse, 4), round(train_mae, 4), round(test_rmse, 4), round(test_mae, 4)


def train_one_epoch(model):
    model.train()
    loss = model()
    optimizer.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_value_(model.parameters(),10)
    optimizer.step()
    return loss.cpu().item()


def train_model(model, par):
    train_rmse_list = []
    train_mae_list = []
    test_rmse_list = []
    test_mae_list = []
    minrmsemae = 9999
    index = 0
    pre_time = 0
    for i in range(args.epochs):
        start_time_pre = datetime.datetime.now()
        train_rmse, train_mae, test_rmse, test_mae = test(model, args.model)
        end_time_pre = datetime.datetime.now()
        time_dlt = (end_time_pre - start_time_pre).microseconds
        pre_time += time_dlt
        train_rmse_list.append(train_rmse)
        train_mae_list.append(train_mae)
        test_rmse_list.append(test_rmse)
        test_mae_list.append(test_mae)
        if (test_rmse + 2 * test_mae) < minrmsemae:
            minrmsemae = test_rmse + 2 * test_mae
            index = i
        loss = train_one_epoch(model)
        if i % 10 == 0:
            print(f'{i}')
            print(f"loss------{loss}")
            print(f"train-----rmse:{train_rmse},mae:{train_mae}")
            print(f"test-----rmse:{test_rmse},mae:{test_mae}")
        if i > 200 and (test_mae + test_rmse) > (test_rmse_list[i - 200] + test_mae_list[i - 200]):
            break

    plt.plot(train_rmse_list, label='train_rmse_list')
    plt.plot(train_mae_list, label='train_mae_list')
    plt.plot(test_rmse_list, label='test_rmse_list')
    plt.plot(test_mae_list, label='test_mae_list')

    plt.plot(index, test_rmse_list[index], 'ks')
    plt.plot(index, test_mae_list[index], 'ks')
    min_rmse = str(test_rmse_list[index])
    min_mae = str(test_mae_list[index])
    plt.annotate(min_rmse, xy=(index, test_rmse_list[index]))
    plt.annotate(min_mae, xy=(index, test_mae_list[index]))
    now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    plt.legend(labels=['train_rmse_list', 'train_mae_list', 'test_rmse_list', 'test_mae_list'])
    if args.untrust_p == 0.0:
        title = 'wiout noise ' + args.model + ' ' + str(par)
    else:
        title = 'noise ' + args.model + ' ' + str(par)
    plt.title(now_time + '\n' + title)
    plt.grid()
    plt.show()
    print(f"min-mae={min_mae}")
    print(f"min-rmse={min_rmse}")
    print('预测时间:', pre_time)
    return float(min_mae), float(min_rmse)


class parses():
    def __init__(self):
        self.model = 'ELF'
        self.attribute = 'rt'

        self.md = 0.025
        # self.untrust_p = 0.0295
        self.untrust_p = 0.1

        self.epochs = 10000
        self.device = 'cuda:3'

        # GCN
        if self.model == 'GCN':
            if self.attribute == 'rt':
                self.lr = 0.01
                self.decay = 0.01
                self.latent_dim = 8
                self.dropout = 1.0
                self.pretrain = False
                self.layer = 2
                self.IOR_MAXI = 10
            else:
                self.lr = 0.01
                self.decay = 0.01
                self.latent_dim = 32
                self.dropout = 1.0
                self.pretrain = False
                self.layer = 3
                self.IOR_MAXI = 10

        # RMF
        elif self.model == 'RMF':
            if self.attribute == 'rt':
                self.lr = 0.01
                self.decay = 0.01
                self.latent_dim = 8
                self.RMF_d = 0.1
                self.RMF_MAXI = 10
            else:
                self.lr = 0.01
                self.decay = 0.01
                self.latent_dim = 8
                self.RMF_d = 0.1
                self.RMF_MAXI = 10

        # LRMF
        elif self.model == 'LRMF':
            if self.attribute == 'rt':
                self.lr = 0.01
                self.decay = 25
                self.latent_dim = 8
                self.ga = 50
                self.RMF_d = 0.1
                self.RMF_MAXI = 10
            else:
                self.lr = 0.01
                self.decay = 0.01
                self.latent_dim = 8
                self.ga = 50
                self.RMF_d = 0.1
                self.RMF_MAXI = 10

        # CMF
        elif self.model == 'CMF':
            if self.attribute == 'rt':
                self.lr = 0.01
                self.decay = 1
                self.latent_dim = 8
                self.gamma = 2
            else:
                self.lr = 0.01
                self.decay = 0.01
                self.latent_dim = 8
                self.gamma = 2

        # PMF
        elif self.model == 'PMF':
            if self.attribute == 'rt':
                self.lr = 0.01
                self.decay = 25
                self.latent_dim = 8

            else:
                self.lr = 0.01
                self.decay = 0.01
                self.latent_dim = 8
        # ELF
        elif self.model == 'ELF':
            if self.attribute == 'rt':
                self.lr = 0.01
                self.decay = 1
                self.latent_dim = 8

            else:
                self.lr = 0.01
                self.decay = 1
                self.latent_dim = 8


if __name__ == '__main__':
    n = 1

    maelist = [0 for i in range(n)]
    rmselist = [0 for i in range(n)]

    # mds = [0.025,0.05,0.075,0.1]
    for i in range(n):
        start_time = datetime.datetime.now()

        args = parses()
        print(f'md:{args.md}')
        print(f'untrust_p:{args.untrust_p}')
        print(f'weight:{args.decay}')
        print(f'latent_dim:{args.latent_dim}')
        dataloader = Dataloader(args)
        # print(dataloader.user_item)
        model, optimizer = get_model()
        par = ' '
        best_mae, best_rmse = train_model(model, par)
        end_time = datetime.datetime.now()
        print(f'md:{args.md}')
        print(f'untrust_p:{args.untrust_p}')
        print(f'weight:{args.decay}')
        print(f'latent_dim:{args.latent_dim}')
        print(f'{i + 1}次耗时' + str(end_time - start_time))
        print("time cost:", (end_time - start_time).seconds, "s")
        print(best_mae)
        print(best_rmse)
        maelist[i] = best_mae
        rmselist[i] = best_rmse
    print(f'{n}轮训练结束')
    print(round(np.mean(maelist), 4))
    print(round(np.mean(rmselist), 4))

# ssh wuziteng@172.31.41.130
