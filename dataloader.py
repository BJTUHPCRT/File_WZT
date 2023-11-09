import torch
import numpy as np

import reputation
import utils
import feature
import os


class Dataloader():
    def __init__(self, args):
        # 构建带权二部图，首先是带权邻接矩阵
        file_rt = open('../data/qos/wsdream_' + args.attribute + '.csv')

        # self.qosdata = np.zeros((339, 4107))
        # for line in file_rt.readlines()[1:]:
        #     index, uid, sid, attr = line.replace('\n', '').split(',')
        #     self.qosdata[int(uid)][int(sid)] = float(attr)

        self.qosdata = np.loadtxt('../data/qos/rtMatrix.txt')

        num_users, num_services = self.qosdata.shape[0], self.qosdata.shape[1]
        self.qosdata = utils.data_process(self.qosdata)
        self.train_mask = np.array(np.random.rand(num_users, num_services) < args.md, dtype=int)
        self.test_mask = np.ones((num_users, num_services)) - self.train_mask
        self.user_item = self.train_mask * self.qosdata
        self.usweight = torch.ones((num_users + num_services, 1)).to(torch.float32).to(args.device)
        self.us_lossweight = torch.ones((num_users, num_users)).to(torch.float32).to(args.device)

        if args.untrust_p > 0:
            untrustuser = utils.noisematrix(self.user_item, args.untrust_p)  #对self.user_item进行原地修改
            if args.model == 'CMF' or args.model == 'PMF' or args.model == 'ELF':
                pass
            else:
                if args.model == 'GCN':
                    rep, sta, avg = reputation.OPM( self.user_item, self.train_mask, args.IOR_MAXI, untrustuser )
                if args.model == 'RMF' or args.model == 'LRMF':
                    rep, sta = reputation.RMF(self.user_item, self.train_mask, args.RMF_d, args.RMF_MAXI, untrustuser)
                R_C = np.append(rep, sta, axis=0)
                self.R_C = torch.from_numpy(R_C).to(torch.float32).to(args.device)
                us_lossweight = np.outer(rep, sta)
                print(us_lossweight)
                self.us_lossweight = torch.from_numpy(us_lossweight).to(torch.float32).to(args.device)
        if args.model == 'GCN':
            self.grpha = utils.getgraph(self.user_item, self.train_mask, rep, avg, args.device)
        
        # self.feature_index = feature.get_feature_indexs()
        self.feature_index = feature.get_feature_indexs2()
        print('data is ready to go')
        # self.feature_graphs['UserSubnet_user'] =\
        #     feature.get_feature_Graph('UserSubnet','user','user',args.device)
        #
        # self.feature_graphs['UserAS_UserSubnet'] = \
        #     feature.get_feature_Graph('UserAS','UserSubnet','user',args.device)
        #
        # self.feature_graphs['UserRegion_UserAS'] = \
        #     feature.get_feature_Graph('UserRegion', 'UserAS', 'user',args.device)
        #
        #
        # self.feature_graphs['ServiceSubnet_service'] = \
        #     feature.get_feature_Graph('ServiceSubnet', 'service','service',args.device)
        #
        # self.feature_graphs['ServiceAS_ServiceSubnet'] = \
        #     feature.get_feature_Graph('ServiceAS','ServiceSubnet','service',args.device)
        #
        # self.feature_graphs['ServiceRegion_ServiceAS'] =\
        #     feature.get_feature_Graph('ServiceRegion', 'ServiceAS', 'service',args.device)
