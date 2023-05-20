import itertools as its
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def relation_matrix_torch(vec1, vec2, e=0):
    dist_matrix = torch.cdist(vec1, vec2, p=1)
    if e == -1:
        return (dist_matrix < 1e-6).float()
    lamb = dist_matrix.mean() * e
    print('Lamb:', lamb)
    relation_matrix = 1 - dist_matrix
    relation_matrix[dist_matrix > lamb] = 0
    return relation_matrix

def relation_matrix_torch_test(vec1, vec2, e=0):
    dist_matrix = torch.cdist(vec1, vec2, p=1)
    if e == -1:
        return (dist_matrix < 1e-6).float()
    relation_matrix = 1 - dist_matrix
    relation_matrix[dist_matrix > e] = 0
    return relation_matrix

class FROD(object):
    def __init__(self):
        pass

    def __make_entropy__(self, X):
        n, m, _ = X.shape
        X = torch.tensor(X, dtype=torch.float32)
        self.M_R_c = torch.zeros((m, n, n), dtype=torch.float32)
        self.FE_x = torch.zeros((n, m), dtype=torch.float32)
        for j in range(m):
            relation_matrix = relation_matrix_torch(X[:, j], X[:, j], self.lambs[j])
            self.M_R_c[j] = relation_matrix
            relation_matrix_sum = relation_matrix.sum(dim=0)
            relation_matrix_sum_x = torch.tile(relation_matrix_sum, (n, 1)) - relation_matrix
            relation_matrix_sum_x = torch.tril(relation_matrix_sum_x, diagonal=-1)[:, :-1] + torch.triu(relation_matrix_sum_x, diagonal=1)[:, 1:]
            self.FE_x[:, j] = -torch.sum(torch.log2(relation_matrix_sum_x / (n - 1)), dim=1) / (n - 1)
        self.FE = -torch.mean(torch.log2(self.M_R_c.mean(dim=1)), dim=1)
        # print(self.FE_x.sum())

    def fit_alpha(self, X, train_y, alpha=1, pruning=None, threshold=0.1, strategy='c'):
        """to be uploaded later"""
        pass


    def fit(self, train_X, train_y, nominals, strategy='c'):
        m = len(nominals)
        self.train_X, self.train_y = train_X, train_y
        lambdas = (0.1 * np.float_power(1.5, np.arange(10))).round(3)
        alphas = [0.01, 0.1, 1, 10, 100]
        results = np.zeros((len(lambdas), len(alphas)))
        fuzzy_appr_acc = np.zeros((len(lambdas), len(alphas), m))
        for i, lamb in enumerate(lambdas):
            for j, alpha in enumerate(alphas):
                self.lambs = np.full(m, lamb)
                self.lambs[nominals] = -1
                self.fit_alpha(train_X, train_y, alpha=alpha, strategy=strategy)
                fuzzy_appr_acc[i, j] = self.fuzzy_appr_acc
                results[i, j] = self.train_auc
            if nominals.sum() == m:
                break

        best_auc = np.max(results)
        best_lamb, best_alpha = np.where(results > best_auc - 1e-6)
        if len(best_lamb) > 1:
            best_lamb = best_lamb[(len(best_lamb)) // 2]
            best_alpha = best_alpha[(len(best_alpha)) // 2]
        else:
            best_lamb, best_alpha = best_lamb[0], best_alpha[0]

        self.lamb = lambdas[best_lamb]
        self.alpha = alphas[best_alpha]
        self.fuzzy_appr_acc = fuzzy_appr_acc[best_lamb, best_alpha]
        self.lambs = np.full(m, self.lamb)
        self.lambs[nominals] = -1
        return self

    def predict_score(self, x=None, strategy='c'):
        if x is not None:
            self.__make_entropy__(x)
            smooth = 1 / len(x)
        else:
            smooth = 1 / len(self.train_X)

        REF = (self.FE_x + 1e-6) / (self.FE + 1e-6) + smooth
        cardinality = self.M_R_c.mean(dim=1)
        FROD = 1 - ((np.sqrt(cardinality) * REF.T) * np.expand_dims(self.fuzzy_appr_acc, axis=1)).mean(axis=0)
        return FROD


def example_FROD():
    from sklearn.preprocessing import minmax_scale
    data = torch.tensor([[0.53, 0.48, 0.5, 0.48, 0.51, 0.52, 0.48, 0.47, 0.53, 0.48],
                         [7,8,7,8,8,7,9,8,9,9],
                         [3,3,2,2,2,3,1,1,1,2]]).T
    label = torch.tensor([1,0,0,0,0,1,0,0,0,0],dtype=torch.float32).reshape(10,1)
    data = torch.tensor(minmax_scale(data).reshape(10,3,1))
    train_X = data[:5]
    train_y = label[:5]
    n, m, _ = train_X.shape
    print(train_X.shape)
    M_R_c = torch.zeros((m,n,n))
    delta = [1,1,-1]
    for i in range(m):
        M_R_c[i] = relation_matrix_torch(train_X[:,i],train_X[:,i],delta[i])
    print(M_R_c)

    M_R_c_all = torch.zeros((m, 10, 10))
    for i in range(m):
        M_R_c_all[i] = relation_matrix_torch(data[:,i],data[:,i],delta[i])
    ODs = M_R_c_all.sum(axis=1).sum(axis=0)
    ODs = minmax_scale(ODs)
    print(ODs)

    M_R_d = relation_matrix_torch(train_y, train_y, -1)
    M_R_d = torch.tensor([[1., 0., 0., 0., 0.], [0., 1., 1., 1., 1.]])
    print(M_R_d)

    low_appr = np.zeros((m, 2, n), dtype=np.float32)
    upp_appr = np.zeros((m, 2, n), dtype=np.float32)
    appr_acc = np.zeros((m, 2), dtype=np.float32)
    for l in range(m):
        M_R_B = M_R_c[l]
        M_R_B_N = 1 - M_R_B
        for i in range(2):
            low_appr[l, i] = torch.min(torch.maximum(M_R_B_N, M_R_d[i]), dim=1)[0]
            upp_appr[l, i] = torch.max(torch.minimum(M_R_B, M_R_d[i]), dim=1)[0]
            appr_acc[l, i] = low_appr[l, i].sum() / upp_appr[l, i].sum()
    appr_acc = appr_acc.sum(axis=1)
    print(low_appr)
    print(upp_appr)
    print(appr_acc)
    # appr_acc = np.array([1,1,1])
    card, FRE = test_make_entropy(X=data[5:])

    FROD = 1 - ((np.sqrt(card) * FRE.T) * np.expand_dims(appr_acc, axis=1)).mean(axis=0)
    print(FROD)

# example_FROD()
