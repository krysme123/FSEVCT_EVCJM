"""
    这个函数是用于 OSR 测试的 FSEVCT & EVCJM !!！
"""
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
from scipy.stats import genextreme as gev
import os
import torch
from copulae import GumbelCopula



def normalized_mahalanobis_distance(samples, center):
    """
    计算所有样本到聚类中心的马氏距离
    :param samples: 样本矩阵, 形状为 [n_samples, n_features]
    :param center: 聚类中心向量, 形状为 [n_features]
    :return: 马氏距离向量, 形状为 [n_samples]
    """
    # 计算协方差矩阵 (考虑样本自由度)
    samples, center = torch.from_numpy(samples), torch.from_numpy(center)
    cov_matrix = torch.cov(samples.T, correction=0)  # 形状 [n_features, n_features]

    # 处理协方差矩阵不可逆问题（使用伪逆）
    if cov_matrix.size(0) == cov_matrix.size(1):
        try:
            inv_cov = torch.linalg.inv(cov_matrix)
        except torch.linalg.LinAlgError:  # 矩阵奇异时用伪逆
            inv_cov = torch.linalg.pinv(cov_matrix)
    else:  # 样本数 < 特征维数，直接用伪逆
        inv_cov = torch.linalg.pinv(cov_matrix)

    # 计算差值向量 [n_samples, n_features]
    delta = samples - center

    # 计算马氏距离: sqrt(delta^T * inv_cov * delta)
    dists = torch.sqrt(torch.einsum('ni,ij,nj->n', delta, inv_cov, delta))
    return dists
    # mean_value = torch.mean(dists)
    # std_value = torch.std(dists)
    # return (dists-mean_value)/std_value


def boundary_fsevct_evcjm(train_features, train_labels, test_features, test_labels, out_features, out_labels,
                           **options):
    print("现在开始使用 FSEVCT & EVCJM 测试模型的开集性能！\n")
    options['boundary_type'] = 'Boundary_FSEVCT_EVCJM'

    c_proto = np.array([train_features[np.argwhere(train_labels == p).ravel()].mean(0)
                        for p in range(options['num_classes'])])

    num_bins = 40       # 这个数值算是个超参数吧，展示极值分布的 bin 的个数
    pdf_x, pdf_y = np.zeros((options['num_classes'], num_bins + 1)), np.zeros((options['num_classes'], num_bins))
    cdf_x, cdf_y = np.zeros((options['num_classes'], num_bins + 1)), np.zeros((options['num_classes'], num_bins))
    x, popt = np.zeros((options['num_classes'], 1000)), np.zeros((options['num_classes'], 2))
    parameters = np.zeros((options['num_classes'], 3))

    gev_image_path = options['save_path'] + '/GEV_image/'
    if not os.path.exists(gev_image_path):
        os.makedirs(gev_image_path)

    train_distance = {}
    train_class_num = 0
    for p in range(options['num_classes']):
        train_distance[str(p)] = normalized_mahalanobis_distance(train_features[np.argwhere(train_labels == p).ravel()],
                                                      c_proto[p, :].reshape(1, -1))
        c, loc, scale = gev.fit(train_distance[str(p)])
        parameters[p, 0], parameters[p, 1], parameters[p, 2] = c, loc, scale
        if p == 0:
            train_class_num = train_distance[str(p)].shape[0]
        else:
            train_class_num = min(train_class_num, train_distance[str(p)].shape[0])




    gev_value = np.zeros((train_class_num, options['num_classes']))
    for p in range(options['num_classes']):
        gev_value[:, p] = gev.cdf(train_distance[str(p)], parameters[p, 0], parameters[p, 1], parameters[p, 2])[0:train_class_num]

    # ######### EVCJM
    cop = GumbelCopula(dim=options['num_classes'])
    cop.fit(gev_value)
    theta_hat = cop.params
    print("当前 EVCJM 模型的 theta 参数为：", theta_hat)

    all_labels = np.concatenate((np.ones(test_labels.shape[0]), np.zeros(out_labels.shape[0])), axis=0)
    y_pred_fsevct = np.zeros((len(options['thresholds']), all_labels.shape[0]))
    y_pred_evcjm = np.zeros((len(options['thresholds']), all_labels.shape[0]))
    y_true = np.concatenate((test_labels, options['num_classes'] * np.ones(out_labels.shape[0])), axis=0)
    all_test_features = np.concatenate((test_features, out_features), axis=0)
    all_test_distance = np.zeros((all_test_features.shape[0], options['num_classes']))
    for i in range(options['num_classes']):
        all_test_distance[:, i] = normalized_mahalanobis_distance(all_test_features, c_proto[i, :])
    options['macro_f1_FSEVCT'] = np.zeros(len(options['thresholds']))
    options['macro_f1_EVCJM'] = np.zeros(len(options['thresholds']))

    func_list = [lambda xx: gev.cdf(xx, parameters[i, 0], parameters[i, 1], parameters[i, 2])
                 for i in range(options['num_classes'])]
    gev_outputs = np.zeros((all_test_distance.shape[0], options['num_classes']))

    for i, func in enumerate(func_list):
        vectorized_func = np.vectorize(func)
        gev_outputs[:, i] = vectorized_func(all_test_distance[:, i])
    all_test_logits_fsevct = gev_outputs
    all_test_logits_evcjm = cop.cdf(gev_outputs)

    evcjm_list = [lambda xx: (-np.log(gev.cdf(xx, parameters[i, 0], parameters[i, 1], parameters[i, 2])))**(theta_hat-1)
                  for i in range(options['num_classes'])]
    evcjm_outputs = np.zeros((all_test_distance.shape[0], options['num_classes']))
    for i, func in enumerate(evcjm_list):
        vectorized_func = np.vectorize(func)
        evcjm_outputs[:, i] =  vectorized_func(all_test_distance[:, i])

    for p in range(len(options['thresholds'])):
        for q in range(len(all_labels)):
            # for r in range(options['num_classes']):
            #     if all_test_logits_fsevct[q] > gev.cdf(all_test_distance[q, r], parameters[r, 0], parameters[r, 1],
            #                                     parameters[r, 2]):
            #         all_test_logits_fsevct[q] = gev.cdf(all_test_distance[q, r], parameters[r, 0], parameters[r, 1],
            #                                      parameters[r, 2])
            #         location = r
            if np.min(all_test_logits_fsevct[q, :]) >= options['thresholds'][p]:
                y_pred_fsevct[p, q] = options['num_classes']
            else:
                y_pred_fsevct[p, q] = np.argmin(all_test_logits_fsevct[q, :])

            if all_test_logits_evcjm[q] >= options['thresholds'][p]:
                y_pred_evcjm[p, q] = options['num_classes']
            else:
                y_pred_evcjm[p, q] =np.argmax(evcjm_outputs[q, :])
        options['macro_f1_FSEVCT'][p] = f1_score(y_true, y_pred_fsevct[p, :], average='macro')
        options['macro_f1_EVCJM'][p] = f1_score(y_true, y_pred_evcjm[p, :], average='macro')

    options['auc_fsevct'] = roc_auc_score(all_labels, -np.min(all_test_logits_fsevct, axis=1))
    fpr, tpr, _ = roc_curve(all_labels, -np.min(all_test_logits_fsevct, axis=1))
    np.save(options['save_path'] + '/' + options['boundary_type'] + '_fpr_FSEVCT', fpr)
    np.save(options['save_path'] + '/' + options['boundary_type'] + '_tpr_FSEVCT', tpr)

    options['auc_evcjm'] = roc_auc_score(all_labels, -all_test_logits_evcjm)
    fpr, tpr, _ = roc_curve(all_labels, -all_test_logits_evcjm)
    np.save(options['save_path'] + '/' + options['boundary_type'] + '_fpr_EVCJM', fpr)
    np.save(options['save_path'] + '/' + options['boundary_type'] + '_tpr_EVCJM', tpr)
