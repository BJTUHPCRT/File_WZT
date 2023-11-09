import torch
import torch.nn as nn
import numpy as np
import utils
import math
import torch.nn.functional as F
import pandas as pd
from sklearn import preprocessing


class MODEL(nn.Module):
    def __init__(self, args, dataloader):
        super(MODEL, self).__init__()
        self.dataloader = dataloader
        self.num_users = self.dataloader.train_mask.shape[0]
        self.num_services = self.dataloader.train_mask.shape[1]

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=args.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_services, embedding_dim=args.latent_dim)

        nn.init.normal_(self.embedding_user.weight, std=0.01)
        nn.init.normal_(self.embedding_item.weight, std=0.01)

        self.args = args
        self.train_mask = torch.from_numpy(self.dataloader.train_mask). \
            to(torch.int).to(self.args.device)
        self.true_qos = torch.from_numpy(self.dataloader.user_item). \
            to(torch.float32).to(self.args.device)
        # print(self.true_qos)

    def getEmbedding(self, test=False):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        return users_emb, items_emb

    def getLoss(self):
        users, items = self.getEmbedding()
        pre_qos = torch.mm(users, items.t())
        pre_qos = torch.mul(pre_qos, self.train_mask)
        true_qos = self.true_qos
        loss = torch.nn.functional.mse_loss(pre_qos, true_qos, reduction='none')
        return loss

    def forward(self):
        loss = self.getLoss()
        loss = torch.sum(loss)
        return loss


class WGCN(MODEL):
    def __init__(self, args, dataloader):
        super(WGCN, self).__init__(args, dataloader)
        '''
        initialization
        '''
        self.usweight = self.dataloader.usweight

        self.us_lossweight = dataloader.us_lossweight

        # self.wsdl_emb = np.loadtxt('wsdl_TFIDF_dim'+str(args.latent_dim))
        self.wsdl_emb = np.loadtxt('wsdl_TFIDF_dim64')
        # self.wsdl_emb = utils.get_item_emb()
        self.wsdl_emb = torch.from_numpy(self.wsdl_emb).to(torch.float32).to(self.args.device)

        # self.embedding_item.weight.data.copy_(self.wsdl_emb)

        '''
        Features!!!
        '''
        self.features = {'user': nn.ModuleList(), 'service': nn.ModuleList()}

        for user_feature_index in self.dataloader.feature_index['user']:
            feature_size = len(np.unique(user_feature_index))
            self.features['user'].append(nn.Embedding(feature_size, args.latent_dim))

        for service_feature_index in self.dataloader.feature_index['service']:
            feature_size = len(np.unique(service_feature_index))
            self.features['service'].append(nn.Embedding(feature_size, args.latent_dim))

        for i in range(len(self.features['user'])):
            nn.init.normal_(self.features['user'][i].weight, std=0.01)
        for i in range(len(self.features['service'])):
            nn.init.normal_(self.features['service'][i].weight, std=0.01)

        self.transfrom = nn.Sequential(
            nn.Linear(in_features=args.latent_dim, out_features=args.latent_dim),
            # nn.BatchNorm1d(args.latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(in_features=args.latent_dim, out_features=args.latent_dim),
            # nn.BatchNorm1d(args.latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(in_features=args.latent_dim, out_features=args.latent_dim),
            # nn.LeakyReLU(),
        )

        # self.transfrom2 = nn.Sequential(
        #     nn.Linear(in_features=args.latent_dim, out_features=args.latent_dim),
        #     # nn.BatchNorm1d(args.latent_dim),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.5),
        #
        #     nn.Linear(in_features=args.latent_dim, out_features=args.latent_dim),
        #     # nn.BatchNorm1d(args.latent_dim),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.5),
        #
        #     nn.Linear(in_features=args.latent_dim, out_features=args.latent_dim),
        #     # nn.LeakyReLU(),
        # )

        self.attention = nn.Sequential(
            nn.Linear(in_features=args.latent_dim * 2, out_features=args.latent_dim),
            # nn.BatchNorm1d(args.latent_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=args.latent_dim, out_features=args.latent_dim),
            # nn.BatchNorm1d(args.latent_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=args.latent_dim, out_features=1),
            # nn.LeakyReLU(),
        )


        self.wsdl_to_vector = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.LeakyReLU(),
            # nn.Dropout(0.5),
            nn.Linear(in_features=32, out_features=16),
            nn.LeakyReLU(),
            # # nn.Dropout(0.5),
            nn.Linear(in_features=16, out_features=8),
        )

        # self.wsdl_to_vector = nn.Sequential(
        #     nn.Linear(in_features=64, out_features=32),
        #     nn.LeakyReLU(),
        #     nn.Linear(in_features=32, out_features=32),
        #     # nn.LeakyReLU(),
        #     # nn.Linear(in_features=32, out_features=32),
        # )

    def get_feature(self, id_emb, entity):
        embs = [id_emb]
        num_feature = len(self.features[entity])
        num_entity = id_emb.shape[0]
        features = []
        if entity == 'user':
            att_scores = torch.zeros(num_entity, num_feature).to(self.args.device)
            for i in range(num_feature):
                feature = self.features[entity][i].weight
                feature_index = self.dataloader.feature_index[entity][i]
                feature_index = torch.LongTensor(feature_index)
                entity_feature = feature[feature_index].to(self.args.device)
                features.append(entity_feature)
                att_scores[:, i] = (
                    torch.squeeze(
                        self.attention(
                            torch.cat([id_emb, entity_feature], dim=1)
                        )
                    )
                )
        else:
            att_scores = torch.zeros(num_entity, num_feature + 1).to(self.args.device)
            for i in range(num_feature + 1):
                if i == num_feature:
                    entity_feature = self.wsdl_to_vector(self.wsdl_emb)
                else:
                    feature = self.features[entity][i].weight
                    feature_index = self.dataloader.feature_index[entity][i]
                    feature_index = torch.LongTensor(feature_index)
                    entity_feature = feature[feature_index].to(self.args.device)
                features.append(entity_feature)
                att_scores[:, i] = (
                    torch.squeeze(
                        self.attention(
                            torch.cat([id_emb, entity_feature], dim=1)
                        )
                    )
                )
        att_scores = torch.softmax(att_scores, dim=1)
        for i in range(num_feature):
            embs.append(
                torch.unsqueeze(att_scores[:, i], 1) * features[i]
            )
        feature_emb = torch.sum(
            torch.stack(embs[1:], dim=1), dim=1
        )
        embs = torch.stack(embs, dim=1)
        embs = torch.sum(embs, dim=1)
        return embs, feature_emb

    def get_feature2(self, id_emb, entity):
        embs = [id_emb]
        num_feature = len(self.features[entity])
        num_entity = id_emb.shape[0]

        features = []
        att_scores = torch.zeros(num_entity, num_feature).to(self.args.device)
        for i in range(num_feature):
            feature = self.features[entity][i].weight
            feature_index = self.dataloader.feature_index[entity][i]
            feature_index = torch.LongTensor(feature_index)
            entity_feature = feature[feature_index].to(self.args.device)
            features.append(entity_feature)
            att_scores[:, i] = (
                torch.squeeze(
                    self.attention(
                        torch.cat([id_emb, entity_feature], dim=1)
                    )
                )
            )

        att_scores = torch.softmax(att_scores, dim=1)
        for i in range(num_feature):
            embs.append(
                torch.unsqueeze(att_scores[:, i], 1) * features[i]
            )
        feature_emb = torch.sum(
            torch.stack(embs[1:], dim=1), dim=1
        )
        embs = torch.stack(embs, dim=1)
        embs = torch.sum(embs, dim=1)
        return embs, feature_emb

    def getEmbedding(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        # users_emb, _ = self.get_feature2(users_emb, 'user')
        # items_emb, _ = self.get_feature2(items_emb, 'service')

        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        for layer in range(self.args.layer):
            all_emb = torch.mm(self.dataloader.grpha, all_emb) / (1 + layer)
            all_emb = self.transfrom(all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        embs = torch.sum(embs, dim=1)
        users_emb, items_emb = torch.split(embs, [self.num_users, self.num_services])
        return users_emb, items_emb

    def forward(self):

        users, items = self.getEmbedding()
        pre_qos = torch.mm(users, items.t())
        # l1_loss
        pre_qos = torch.mul(pre_qos, self.train_mask)
        loss = torch.nn.functional.l1_loss(pre_qos, self.true_qos, reduction='none')
        # loss = torch.nn.functional.smooth_l1_loss(pre_qos, self.true_qos, reduction='none')
        # loss = torch.nn.functional.mse_loss(pre_qos, self.true_qos, reduction='none')
        loss = torch.mul(self.us_lossweight, loss)

        loss = torch.sum(loss)

        return loss


class LRMF(MODEL):
    def __init__(self, args, dataloader):
        super(LRMF, self).__init__(args, dataloader)
        self.us_lossweight = dataloader.us_lossweight

    def get_regionloss(self):
        user_emb = self.embedding_user.weight
        region_index = self.dataloader.feature_index['user'][0]
        regions = len(np.unique(region_index))  # 一共有多少个region
        sum_regionloss = 0
        for region in range(regions):
            user_region = user_emb[region_index == region]  # 得到当前region下的用户集合
            if user_region.shape[0] == 1:
                continue
            user_region_sum = torch.unsqueeze(torch.sum(user_region, dim=0), 0)
            regionloss = user_region - (user_region_sum - user_region) / (user_region.shape[0] - 1)
            regionloss = torch.abs(regionloss)
            regionloss = torch.sum(regionloss)
            sum_regionloss += regionloss
        return sum_regionloss

    def forward(self):
        loss = self.getLoss()
        loss = torch.mul(self.us_lossweight, loss)
        loss = torch.sum(loss)
        regionloss = self.get_regionloss()
        # print(f"loss={loss}")
        # print(f"regionloss={regionloss}")
        # return loss
        return loss + self.args.ga * regionloss


class RMF(MODEL):
    def __init__(self, args, dataloader):
        super(RMF, self).__init__(args, dataloader)
        self.us_lossweight = dataloader.us_lossweight

    def forward(self):
        loss = self.getLoss()
        loss = torch.mul(self.us_lossweight, loss)
        loss = torch.sum(loss)
        return loss




class CMF(MODEL):
    def __init__(self, args, dataloader):
        super(CMF, self).__init__(args, dataloader)

    def forward(self):
        loss = self.getLoss()
        gamma_2 = self.args.gamma ** 2
        loss = torch.log(1 + loss / gamma_2)
        loss = torch.sum(loss)
        return loss
