import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model import FROD
import pickle as pkl

def get_train_data(data, label, train_ratio, seed):
    n, m = data.shape
    data_label = np.concatenate([data, np.arange(n).reshape((n, 1))], axis=1)
    data_negative = data_label[label != 1]
    data_positive = data_label[label == 1]
    n_train_neg = round(len(data_negative) * train_ratio)
    n_train_pos = round(len(data_positive) * train_ratio)
    if n_train_pos <1:
        print('Training without labeled anomaly.')
    n_train = n_train_pos + n_train_neg
    n_train, n_train_neg, n_train_pos = n_train, n_train_neg, n_train_pos

    np.random.seed(seed)
    np.random.shuffle(data_negative)
    np.random.seed(seed)
    np.random.shuffle(data_positive)

    train, test = data_negative[:n_train_neg], data_negative[n_train_neg:]
    train_x_n, test_x_n = train[:, :-1], test[:, :-1]
    train_y_n, test_y_n = train[:, -1], test[:, -1]

    train, test = data_positive[:n_train_pos], data_positive[n_train_pos:]
    train_x_p, test_x_p = train[:, :-1], test[:, :-1]
    train_y_p, test_y_p = train[:, -1], test[:, -1]
    train_x = np.concatenate([train_x_p, train_x_n])
    train_y = np.concatenate([np.ones_like(train_y_p, dtype=np.int32), np.zeros_like(train_y_n, dtype=np.int32)])
    test_x = np.concatenate([test_x_p, test_x_n])
    test_y = np.concatenate([np.ones_like(test_y_p, dtype=np.int32), np.zeros_like(test_y_n, dtype=np.int32)])
    return train_x, train_y, test_x, test_y


result_file_name = 'results_FROD.csv'
open(result_file_name, 'w').write('dataset,model,lamb,n_train,seed,auc,pr\n')

dataset_list = np.array(os.listdir('data'))
for dataset in dataset_list:
    d = np.load('data/'+dataset)
    data = d['X']
    valids = np.std(data, axis=0) > 1e-6
    data = data[:, valids]
    n, m = data.shape
    label = d['y']
    nominals = d['nominals'][valids]
    print("Data shape:{}  # Outlier:{} # Nominals:{}".format((n, m), label.sum(),nominals.sum()))

    for n_train in [.02, .04, .08]:
        for seed in range(2023, 2033, 1):
            train_X, train_y, test_X, test_y = get_train_data(data, label, n_train, seed)
            train_X, test_X = train_X[:, :, np.newaxis], test_X[:, :, np.newaxis]
            model = SSOD().fit(train_X, train_y, nominals)

            out_scores = model.predict_score(test_X)
            auc = roc_auc_score(test_y, out_scores)
            pr = average_precision_score(y_true=test_y, y_score=out_scores, pos_label=1)
            print(dataset[:-4], model.alpha, model.lamb, '\t', n_train, seed, auc, pr)

            scores = [dataset[:-4], 'FROD', str(model.lamb), str(n_train), str(seed), str(auc)[:8], str(pr)[:8]]
            open(result_file_name, 'a').write(','.join(scores) + '\n')



