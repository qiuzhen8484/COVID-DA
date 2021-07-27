import logging
import os
import time
import sklearn.metrics as metrics
from sklearn.metrics import f1_score


def log():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    log_dir_path = os.path.join(os.getcwd(), 'log')
    if not os.path.exists(log_dir_path):
        os.mkdir(log_dir_path)
    log_name = os.path.join(log_dir_path, time.strftime('%Y%m%d%H%M') + '.txt')
    handler = logging.FileHandler(log_name, mode='w')
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger


def analyse(gt_list, p_list, logger, prob=True):
    if prob:
        AUROC = metrics.ranking.roc_auc_score(gt_list, p_list)
        logger.info('AUROC: %.4f' % AUROC)

        p_list[p_list >= 0.5] = 1
        p_list[p_list < 0.5] = 0

    t_open, f_narrow, f_open, t_narrow = metrics.confusion_matrix(gt_list, p_list).ravel()
    logger.info('true_negative: %s ; false_positive: %s ; false_negative: %s ; true_positive: %s' % (t_open, f_narrow,
                                                                                                     f_open, t_narrow))

    F1 = f1_score(gt_list, p_list)
    accuracy = (t_narrow+t_open) / (t_narrow+t_open+f_narrow+f_open)
    precision = t_narrow / (t_narrow+f_narrow)
    recall = t_narrow / (t_narrow+f_open)
    logger.info('F1: %.4f' % F1)
    logger.info('precision: %.4f ; recall: %.4f' % (precision, recall))
    return F1

