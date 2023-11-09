import numpy as np
import torch


def filterqos(user_item):
    data = user_item.copy()
    temp_qos = []
    for i in range(len(data)):
        for j in range(len(data[0])):
            if data[i][j] > 0:
                temp_qos.append(data[i][j])
    temp_qos.sort()
    low_qos = temp_qos[int(len(temp_qos) * 0.3)]
    high_qos = temp_qos[-int(len(temp_qos) * 0.3)]
    filter_mask = np.ones((339, 4107))
    for i in range(len(data)):
        for j in range(len(data[0])):
            if 0 < data[i][j] < low_qos:
                filter_mask[i][j] = -1
            elif low_qos < data[i][j] < high_qos:
                filter_mask[i][j] = 0
    return filter_mask


def get_item_emb():
    file = open('./txtfile/doc2vec_dim8.txt', 'r')
    items = []
    for item in file.readlines()[1:1 + 4107]:
        item_emb = item.split(' ')[1:]
        item_emb = list(map(float, item_emb))
        items.append(item_emb)
    items = np.array(items)
    return items


def data_process(a):
    a = a + 0.001
    return a


def data_deprocess(a):
    return a - 0.001


def refineqos(user_item, mask, rep, avg):
    dis = avg-user_item
    refine = (1-rep)*dis
    return user_item+refine*mask
    # dis = np.abs(user_item-avg) #记录用户和评分均值的差异


def getgraph(user_item, mask, rep,avg, device):

    row = mask.shape[0]
    col = mask.shape[1]
    userdegree = np.sum(mask, axis=1)  # 求出每行用户的度
    itemdegree = np.sum(mask, axis=0)  # 求出每列item的度
    degree = np.append(userdegree, itemdegree, axis=0) ** (-0.5)
    D = np.diag(degree)

    #1.A
    # adj = mask
    # print('adj=A')

    #2.Q
    # adj = user_item
    # print('adj=Q')

    #3.A*R
    adj = mask * rep
    print('adj=A*R')

    #4.Q*R
    # adj = user_item*rep
    # print('adj=Q*R')

    #5.refine_Q
    # adj = refineqos(user_item, mask, rep, avg)
    # print('adj=refine_Q')

    A = np.zeros((row + col, row + col))
    A[:row, row:] = adj
    A[row:, :row] = adj.T
    grpha = np.dot(np.dot(D, A), D)
    grpha = torch.from_numpy(grpha).to(torch.float32).to(device)
    return grpha


def noisematrix(matrix, p):
    num_users = matrix.shape[0]
    num_services = matrix.shape[1]
    temp_matrix = matrix[matrix > 0]
    num = int(np.round( num_users*p ))
    untrustuser = np.zeros( num_users )
    untrustuser[:num] = 1
    np.random.shuffle(untrustuser)
    untrustuser = np.array(untrustuser,dtype=int)
    for i in range(num_users):
        if untrustuser[i] == 1:
            for j in range(num_services):
                if matrix[i][j] > 0:
                    a = np.random.choice(temp_matrix) * 1.5
                    matrix[i][j] = a
    return untrustuser


def noisematrix2(matrix, p):
    temp_matrix = matrix[matrix > 0]
    num = int(np.round(339*p))
    untrustuser = np.zeros(339)
    untrustuser[:num]=1
    np.random.shuffle(untrustuser)
    untrustuser = np.array(untrustuser,dtype=int)
    for i in range(339):
        if untrustuser[i] == 1:
            for j in range(4107):
                if matrix[i][j] > 0:
                    a = np.random.choice(temp_matrix)
                    matrix[i][j] = a
    return matrix, untrustuser