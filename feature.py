import pandas as pd
from sklearn import preprocessing
import numpy as np
import utils
from scipy.spatial import distance


def get_feature_Graph(feature1, feature2, node, device):
    FeatureEncoder = preprocessing.LabelEncoder()
    if node == 'user':
        feature = pd.read_csv('../data/qos/user_feature.csv', sep='::', engine='python')
        feature2_label = range(339)
    else:
        feature = pd.read_csv('../data/qos/service_feature.csv', sep='::', engine='python')
        feature2_label = range(4107)
    for index, row in feature.iteritems():
        if index == feature1:
            feature1_label = FeatureEncoder.fit_transform(feature[index])
        if index == feature2:
            feature2_label = FeatureEncoder.fit_transform(feature[index])

    num_feature1 = len(np.unique(feature1_label))
    num_feature2 = len(np.unique(feature2_label))
    print(num_feature1)
    print(num_feature2)
    R = np.zeros((num_feature1, num_feature2))
    for i in range(len(feature1_label)):
        row = feature1_label[i]
        column = feature2_label[i]
        R[row][column] = 1
    return utils.getgraph(R, R, device)


# get_feature_Graph('ServiceRegion','ServiceAS','service')
#
#     self.U_Feature.append(nn.Embedding(feature_size, dim))
#     self.U_index_feature.append(index_feature)
# service_feature = pd.read_csv('../data/qos/service_feature.csv', sep='::')
# self.S_Feature = nn.ModuleList()
# self.S_index_feature = []
# for index, row in service_feature.iteritems():
#     index_feature = FeatureEncoder.fit_transform(service_feature[index])
#     feature_size = len(np.unique(index_feature))
#     self.S_Feature.append(nn.Embedding(feature_size, dim))
#     self.S_index_feature.append(index_feature)
def get_feature_indexs():
    FeatureEncoder = preprocessing.LabelEncoder()

    user_feature = pd.read_csv('../data/qos/user_feature.csv', sep='::')
    user_feature_idnexs = []
    for index, row in user_feature.iteritems():
        feature_index = FeatureEncoder.fit_transform(user_feature[index])
        user_feature_idnexs.append(feature_index)

    service_feature = pd.read_csv('../data/qos/service_feature.csv', sep='::')
    service_feature_idnexs = []
    for index, row in service_feature.iteritems():
        feature_index = FeatureEncoder.fit_transform(service_feature[index])
        service_feature_idnexs.append(feature_index)
    return {'user':user_feature_idnexs, 'service':service_feature_idnexs}

def get_feature_indexs2():
    FeatureEncoder = preprocessing.LabelEncoder()

    user_feature = pd.read_csv('../data/qos/userlist.txt', sep='	')
    user_feature_idnexs = []
    for index, row in user_feature.iteritems():
        print(index)
        print(user_feature[index])
        feature_index = FeatureEncoder.fit_transform(user_feature[index])
        user_feature_idnexs.append(feature_index)

    service_feature = pd.read_csv('../data/qos/wslist.txt', sep='	', encoding= 'ISO-8859-1')
    service_feature_idnexs = []
    for index, row in service_feature.iteritems():
        print(index)
        print(service_feature[index])
        feature_index = FeatureEncoder.fit_transform(service_feature[index])
        service_feature_idnexs.append(feature_index)
    return {'user':user_feature_idnexs, 'service':service_feature_idnexs}
