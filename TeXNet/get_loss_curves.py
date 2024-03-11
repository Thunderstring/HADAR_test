
import numpy as np
import os
from glob import glob
from tensorflow.python.summary.summary_iterator import summary_iterator
import matplotlib.pyplot as plt

FOLDS = [0, 1, 2, 3, 4]
AVAILABLE_GRAPHS = ["loss", "loss1", "loss_T", "loss_e", "loss_v", "loss_S", "train_miou",
                    "val_miou", "val_loss", "val_loss_e", "val_loss_T", "val_loss_v", "val_loss_S"]
GRAPH_TITLES = ["loss", "loss1", "loss_T", "loss_e", "loss_v", "loss_S", "train_miou",
                    "val_miou", "val_loss", "val_loss_e", "val_loss_T", "val_loss_v", "val_loss_S"]

KEYS_TO_PLOT = ["loss_T", 'val_loss_T', 'val_loss']
COLORS = ["r", "g", "b", "m", "k"]
N = 21
smooth_filter = (1./N)*np.ones(N)

# for key in KEYS_TO_PLOT:
#     for fold in FOLDS:
#         # data = {str(k): [] for k in FOLDS}
#         data = []
#         log_folder = f"./supervised_synexp_r50_fold{fold}_new/lightning_logs/version_1"
#         log_file = glob(os.path.join(log_folder, "events.out.tfevents.*"))[-1]
#         for summary in summary_iterator(log_file):
#             for v in summary.summary.value:
#                 if v.tag == key:
#                     # data[fold].append(v.simple_value)
#                     data.append(v.simple_value)
#         # plt.plot(data[fold], label=f"Fold {fold}")
#         if "val" in key:
#             plt.plot(data, label=f"Fold {fold}", c=COLORS[FOLDS.index(fold)])
#         else:
#             plt.plot(data, alpha=0.1, c=COLORS[FOLDS.index(fold)])
#             data_ = np.concatenate((data[0]*np.ones(N//2), np.array(data), data[-1]*np.ones(N//2)))
#             smooth_data = np.convolve(data_, smooth_filter, mode='valid')
#             plt.plot(smooth_data, label=f"Fold {fold}", c=COLORS[FOLDS.index(fold)])
    
#     plt.grid()
#     plt.ylabel("Value")
#     plt.xlabel("Steps")
#     plt.legend()
#     plt.title(GRAPH_TITLES[AVAILABLE_GRAPHS.index(key)])
#     plt.savefig(f"fig_{key}.png")
#     plt.clf()
for key in KEYS_TO_PLOT:
    fold = 0
    # data = {str(k): [] for k in FOLDS}
    data = []
    log_folder = f"./supervised_crop_10000epoch/lightning_logs/version_1"
    log_file = glob(os.path.join(log_folder, "events.out.tfevents.*"))[-1]
    for summary in summary_iterator(log_file):
        for v in summary.summary.value:
            if v.tag == key:
                # data[fold].append(v.simple_value)
                data.append(v.simple_value)
    # plt.plot(data[fold], label=f"Fold {fold}")
    if "val" in key:
        plt.plot(data, label=f"Fold {fold}", c=COLORS[FOLDS.index(fold)])
    else:
        plt.plot(data, alpha=0.1, c=COLORS[FOLDS.index(fold)])
        data_ = np.concatenate((data[0]*np.ones(N//2), np.array(data), data[-1]*np.ones(N//2)))
        smooth_data = np.convolve(data_, smooth_filter, mode='valid')
        plt.plot(smooth_data, label=f"Fold {fold}", c=COLORS[FOLDS.index(fold)])

plt.grid()
plt.ylabel("Value")
plt.xlabel("Steps")
plt.legend()
plt.title(GRAPH_TITLES[AVAILABLE_GRAPHS.index(key)])
plt.savefig(f"fig_{key}.png")
plt.clf()
