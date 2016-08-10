__author__ = 'joerg'

import numpy as np
from sklearn import metrics

#import matplotlib.pyplot as plt

class boundary_evaluation:

    def __init__(self):
        pass

    @staticmethod
    def validation_batches(output, target, mask, tolerance):

        n_batches = output.__len__()
        batch_size = output[0].shape[1]

        threshold = np.array([0.5]) #np.arange(0,1,0.1)

        tp = np.zeros(threshold.shape)
        tn = np.zeros(threshold.shape)
        fp = np.zeros(threshold.shape)
        fn = np.zeros(threshold.shape)

        if tolerance == 1:
            tol_range = np.array([0,-1,1])
        elif tolerance == 2:
            tol_range = np.array([0,-1,1,-2,2])
        else:
            assert False
        # deep copy beacuase i change the target because of tolerance adjustment
        #target = np.copy(target_org)

        # for i in range(n_batches):
        #     for t in range(threshold.shape[0]):
        #         for b in range(batch_size):
        #             #out_signal = output[i][:,b,1] >= threshold[t]
        #             out_signal = output[i][:,b,1]
        #             out_signal = (out_signal > np.roll(out_signal,1)) & (out_signal > np.roll(out_signal,-1))  & (out_signal > threshold[t])
        #             for s in range(output[i].shape[0]):
        #                 if mask[i][s,b] == 1:
        #                     if out_signal[s] == 1:
        #                         if (target[i][ max(s-tolerance,0):min(s+tolerance+1,output[i].shape[0])  ,b,1].sum() >= 1):
        #                         #for tol in tol_range:
        #                         #    if target[i][np.clip(s+tol,0,output[i].shape[0]-1), b, 1] == 1 and signal_found == 0:
        #                         #    signal_found = 1
        #                             tp[t] += 1
        #                         #if signal_found == 0:
        #                         else:
        #                             fp[t] += 1
        #
        #                     else:
        #                         if target[i][ s,b,1] == 0 or (out_signal[ max(s-tolerance,0):min(s+tolerance+1,output[i].shape[0])].sum() >= 1):
        #                             tn[t] += 1
        #                         else:
        #                             fn[t] += 1

        out_signal = []
        tar_signal = []

        for i in range(n_batches):
            for t in range(threshold.shape[0]):
                for b in range(batch_size):
                    #out_signal = output[i][:,b,1] >= threshold[t]
                    out_signal_ = output[i][0:int(mask[i][:,b,0].sum()),b,1]
                    out_signal_ = (out_signal_ > threshold[t]) & (out_signal_ > np.roll(out_signal_,1)) & (out_signal_ > np.roll(out_signal_,-1))
                    tar_signal_ = np.copy(target[i][0:int(mask[i][:,b,0].sum()),b,1])
                    help_signal = np.zeros([out_signal_.__len__()])

                    # plt.subplot(2,1,1)
                    # plt.plot(output[i][0:int(mask[i][:,b,0].sum()),b,1])
                    # plt.plot(out_signal_)
                    # plt.subplot(2,1,2)
                    # plt.plot(tar_signal_, c='g')


                    for s in range(out_signal_.shape[0]):
                        if out_signal_[s] == 1:
                            signal_found = 0
                            for tol in tol_range:
                                if tar_signal_[np.clip(s+tol,0,tar_signal_.shape[0]-1)] == 1 and signal_found == 0 and help_signal[np.clip(s+tol,0,tar_signal_.shape[0]-1)] == 0:
                                    signal_found = 1
                                    tar_signal_[np.clip(s+tol,0,tar_signal_.shape[0]-1)] = 0
                                    tar_signal_[s] = 1
                                    help_signal[s] = 1

                    # plt.plot(tar_signal_*0.5, c='r')
                    # plt.show()

                    out_signal.append(out_signal_)
                    tar_signal.append(tar_signal_)

        # print(out_signal.__len__())
        # print(out_signal[0].shape[0])
        # print(out_signal[0])
        # print(tar_signal[0])

        # todo different tolerance implementation

        for i in range(out_signal.__len__()):
            for s in range(out_signal[i].shape[0]):
                if out_signal[i][s] == 1:
                    if tar_signal[i][s] == 1:
                        tp[t] += 1
                    else:
                        fp[t] += 1
                else:
                    if tar_signal[i][s] == 1:
                        fn[t] += 1
                    else:
                        tn[t] += 1





        # calculate error measurements
        accuracy = np.zeros(threshold.shape)
        precision = np.zeros(threshold.shape)
        recall = np.zeros(threshold.shape)
        f1_score = np.zeros(threshold.shape)
        for t in range(threshold.shape[0]):
            accuracy[t] = np.true_divide((tp[t] + tn[t]), (tp[t]+tn[t]+fn[t]+fp[t])+ 1e-8 )
            precision[t] = np.true_divide(tp[t],(tp[t]+fp[t])+ 1e-8)
            recall[t] = np.true_divide(tp[t],(tp[t]+fn[t])+ 1e-8)
            f1_score[t] = np.true_divide((2*tp[t]),(2*tp[t]+fp[t]+fn[t])+ 1e-8)
        area_uncer_curve = 0#metrics.auc(precision, recall)ll)



        t = np.where(threshold == 0.5)
        # print("Accuracy:  ",accuracy[t])
        # print("Percision: ",precision[t])
        # print("Recall:    ",recall[t] )
        # print("F1 Score:  ",f1_score[t])
        # print(tp[t])
        # print(tn[t])
        # print(fp[t])
        # print(fn[t])

        # print("AUC:  ",area_uncer_curve)

        # t = np.where(f1_score == f1_score.max())
        # print("best threshold:", threshold[t])
        # print("Accuracy:  ",accuracy[t])
        # print("Percision: ",precision[t])
        # print("Recall:    ",recall[t] )
        # print("F1 Score:  ",f1_score[t])


        return accuracy[t],f1_score[t],area_uncer_curve



