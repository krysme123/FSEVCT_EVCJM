"""
    这个函数是用于 OSR 测试的 极值边界法！
"""
import numpy as np
from Auxiliary.create_logfile import create_boundary_analysis_logfile
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
from scipy.stats import genextreme as gev
from scipy import stats
import os


def boundary_extreme_value(train_features, train_labels, test_features, test_labels, out_features, out_labels,
                           **options):
    print("现在开始使用极值边界测试模型的开集性能！\n")
    options['boundary_type'] = 'Boundary_extreme_value'

    c_proto = np.array([train_features[np.argwhere(train_labels == p).ravel()].mean(0)
                        for p in range(options['num_classes'])])

    num_bins = 50       # 这个数值算是个超参数吧，展示极值分布的 bin 的个数
    pdf_x, pdf_y = np.zeros((options['num_classes'], num_bins + 1)), np.zeros((options['num_classes'], num_bins))
    cdf_x, cdf_y = np.zeros((options['num_classes'], num_bins + 1)), np.zeros((options['num_classes'], num_bins))
    x, popt = np.zeros((options['num_classes'], 1000)), np.zeros((options['num_classes'], 2))
    parameters = np.zeros((options['num_classes'], 3))

    gev_image_path = options['save_path'] + '/GEV_image/'
    if not os.path.exists(gev_image_path):
        os.makedirs(gev_image_path)

    train_distance = {}
    for p in range(options['num_classes']):
        train_distance[str(p)] = pairwise_distances(
            train_features[np.argwhere(train_labels == p).ravel()], c_proto[p, :].reshape(1, -1),
            metric="euclidean", n_jobs=-1).ravel()

        plt.figure()
        c, loc, scale = gev.fit(train_distance[str(p)])
        parameters[p, 0], parameters[p, 1], parameters[p, 2] = c, loc, scale
        pdf_y[p, :], pdf_x[p, :], _ = plt.hist(train_distance[str(p)], histtype='barstacked', density=True,
                                               bins=num_bins, edgecolor='black')
        plt.plot(pdf_x[p, :], gev.pdf(pdf_x[p, :], c, loc, scale),
                 label=r'GEV_PDF: $\xi={:.3f}, \alpha={:.3f}, \beta={:.3f}$'.format(c, loc, scale))
        plt.xlabel(r'$||\Theta(x)-O||_2$')
        plt.ylabel('Probability density')
        plt.legend()
        plt.savefig(gev_image_path + 'GEV_PDF_{}.pdf'.format(p))

        plt.figure()
        cdf_y[p, :], cdf_x[p, :], _ = plt.hist(train_distance[str(p)], histtype='barstacked', density=True,
                                               cumulative=True, bins=num_bins, edgecolor='black')
        statistic, pvalue = stats.ks_2samp(cdf_y[p, :], gev.cdf(cdf_x[p, :], c, loc, scale))
        x[p, :] = np.linspace(cdf_x[p, 0], cdf_x[p, -1], 1000)
        plt.plot(x[p, :], gev.cdf(x[p, :], c, loc, scale), label='GEV_CDF: $P={:.3f}$'.format(pvalue))
        plt.xlabel(r'$||\Theta(x)-O||_2$')
        plt.ylabel('Probability')
        plt.legend()
        plt.savefig(gev_image_path + 'GEV_CDF_{}.pdf'.format(p))

        plt.close()

    all_labels = np.concatenate((np.ones(test_labels.shape[0]), np.zeros(out_labels.shape[0])), axis=0)
    all_test_logits = np.ones(len(all_labels))
    y_pred = np.zeros((len(options['thresholds']), all_labels.shape[0]))
    y_true = np.concatenate((test_labels, options['num_classes'] * np.ones(out_labels.shape[0])), axis=0)
    all_test_features = np.concatenate((test_features, out_features), axis=0)
    all_test_distance = pairwise_distances(all_test_features, c_proto, metric='euclidean', n_jobs=-1)
    options['macro_f1'] = np.zeros(len(options['thresholds']))

    location = None
    for p in range(len(options['thresholds'])):
        for q in range(len(all_labels)):
            for r in range(options['num_classes']):
                if all_test_logits[q] > gev.cdf(all_test_distance[q, r], parameters[r, 0], parameters[r, 1],
                                                parameters[r, 2]):
                    all_test_logits[q] = gev.cdf(all_test_distance[q, r], parameters[r, 0], parameters[r, 1],
                                                 parameters[r, 2])
                    location = r
            if all_test_logits[q] >= options['thresholds'][p]:
                y_pred[p, q] = options['num_classes']
            else:
                y_pred[p, q] = location
        options['macro_f1'][p] = f1_score(y_true, y_pred[p, :], average='macro')

    options['auc'] = roc_auc_score(all_labels, -all_test_logits)
    fpr, tpr, _ = roc_curve(all_labels, -all_test_logits)
    np.save(options['save_path'] + '/' + options['boundary_type'] + '_fpr', fpr)
    np.save(options['save_path'] + '/' + options['boundary_type'] + '_tpr', tpr)

    create_boundary_analysis_logfile(**options)
