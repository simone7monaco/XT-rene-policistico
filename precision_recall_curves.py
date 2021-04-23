import cv2
import argparse
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from scipy.stats import logistic
from sklearn.metrics import (
    PrecisionRecallDisplay,
    jaccard_score,
    precision_recall_curve,
)
from tqdm import tqdm

from scipy.sparse import csr_matrix

from utils import (
    get_samples,
    object_from_dict,
    state_dict_from_disk,
    tensor_from_rgb_image,
)
from time import time

def get_args():
    parser = argparse.ArgumentParser(description=f'P-R curve for remove-thr postprocessing')
    parser.add_argument('exp', nargs='*', type=int, default=[4], help="Exp between 1~4")
    args = parser.parse_args()
    return args

#sd ={} with NAMES
# names are lists of cysts with {area, type}
# type: {db,ds,mb,ms,wb,ws,ob,os}

def missed_wrong_cysts(gt, pred, small_thresh=159, remove_thr=70):
    detected_big = 0
    missed_big = 0
    detected_small = 0
    missed_small = 0
    total_small = 0
    total_big = 0
    overcounted_big = 0
    overcounted_small = 0    

    gt_contours, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours(pred.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    gt_seps = np.array([csr_matrix(cv2.fillPoly(np.zeros_like(gt), pts=[c], color=(1))) for c in gt_contours if c.size > 2], dtype=object)
    
    pred_seps = np.array([csr_matrix(cv2.fillPoly(np.zeros_like(gt), pts=[c], color=(1))) for c in pred_contours if cv2.contourArea(c) > remove_thr and c.size > 2], dtype=object)

    detect_miss_list = np.array([False for _ in pred_seps])
    
    n_rem = len(pred_contours) - len(pred_seps)

    for single_gt, c in zip(gt_seps, gt_contours):
        area = cv2.contourArea(c)
        if area > small_thresh:
            total_big += 1
            curr_detection = np.array([single_gt.multiply(sing_p).count_nonzero() for sing_p in pred_seps], dtype=bool)
            if curr_detection.any():
                detected_big += 1
                overcounted_big += sum(curr_detection) - 1
                # -> the other to #overcounted [assoluto] -- diviso per dimensione
            else:
                missed_big += 1
        else:
            total_small += 1
            curr_detection = np.array([single_gt.multiply(sing_p).count_nonzero() for sing_p in pred_seps], dtype=bool)
            if curr_detection.any():
                detected_small += 1
                overcounted_small += sum(curr_detection) - 1
            else:
                missed_small += 1
        
        detect_miss_list += curr_detection
    
    if len(pred_seps) == 0:
        return (
            total_small,
            detected_small,
            missed_small,
            0,
            overcounted_small,
            total_big,
            detected_big,
            missed_big,
            0,
            overcounted_big,
            n_rem
        )

    wrong_contours = [c for c in pred_contours if cv2.contourArea(c) > remove_thr]
    wrong_contours = [c for c,t in zip(wrong_contours, ~detect_miss_list) if t]
    wrong_big = sum([True for c in wrong_contours if cv2.contourArea(c) > small_thresh])
    wrong_small = sum(~detect_miss_list) - wrong_big
    
    # NON NECESSARIAMENTE: una ciste che sta su due vicine vale per 1 (caso raro)
#     assert sum((detected_big, detected_small, missed_big, missed_small)) == len(gt_contours), f"{m_p.name}: {detected_big} + {detected_small} + {missed_big} + {missed_small} != {len(gt_contours)}"
    
    # NON NECESSARIAMENTE: una ciste può essere overcount di più corrected
#     assert sum((detected_big, detected_small, wrong_big, wrong_small, overcounted_big, overcounted_small)) == len(pred_seps), f"{m_p.name}: {detected_big} + {detected_small} + {wrong_big} + {wrong_small} + {overcounted_big} + {overcounted_small} != {len(pred_seps)}"

    return (
        total_small,
        detected_small,
        missed_small,
        wrong_small,
        overcounted_small,
        total_big,
        detected_big,
        missed_big,
        wrong_big,
        overcounted_big,
        n_rem
    )


args = get_args()
n_ex = args.exp[0]

mask_PATH = Path(".") / "DATASET/masks/all_oldnames/"
pred_PATH = Path(".") / f"cv_perexp/exp{n_ex}/orig_names/"
# pred_PATH = Path(".") / f"results/seed_1989_cv/"

out_PATH = Path(".") / f"curves/exp_{n_ex}"
out_PATH.mkdir(exist_ok=True, parents=True)
print(f"Experiment n° {n_ex}\n")


# samples = get_samples(
#     "/datadisk/rene-policistico/CV/valid_imgs_4_1989/",
#     "/datadisk/rene-policistico/CV/valid_masks_4_1989/",
# )

# device = torch.device("cuda", 0)
# model = {
#     "type": "segmentation_models_pytorch.UnetPlusPlus",
#     "encoder_name": "resnet50",
#     "classes": 1,
#     "encoder_weights": "imagenet",
# }
# model = object_from_dict(model)
# model = model.to(device)

# corrections = {"model.": ""}
# state_dict = state_dict_from_disk(
#     file_path="/datadisk/rene-policistico/logs_cv/UnetPlusPlus/fold_4_1989/epoch=79-val_iou=0.6671.ckpt",
#     rename_in_layers=corrections,
# )

# model.load_state_dict(state_dict)

# transform = albu.augmentations.transforms.Normalize()

# dataloader = DataLoader(
#     SegmentationDataset(samples, transform, length=None),
#     batch_size=1,
#     num_workers=4,
#     shuffle=False,
#     pin_memory=True,
#     drop_last=True,
# )

# model.eval()

# loader = tqdm(dataloader)

# thresholds = list(np.arange(0, 159, 3)) # for remove_thr
thresholds = np.arange(80, 350, 10) # for StoB
colors = [
    "navy",
    "deeppink",
    "orangered",
    "olivedrab",
    "royalblue",
    "lightseagreen"
]

TPss_list = []
FPss_list = []
FNss_list = []
TPsb_list = []
FPsb_list = []
FNsb_list = []
OVb = []
OVs = [] 
N_rem = 0

EPSILON = 1e-15

samples_dict = {}

t_start = time()
it = tqdm(thresholds)
for thresh in it:
    it.set_description(f"Removed [{N_rem}]", refresh=True)
    TPs_small = 0
    FPs_small = 0
    FNs_small = 0
    TPs_big = 0
    FPs_big = 0
    FNs_big = 0
    ovb = 0
    ovs = 0
    N_rem = 0

    for m_p in pred_PATH.glob("*.png"):
        
        mask = cv2.imread(
            str(m_p),
            cv2.IMREAD_GRAYSCALE,
        )
        gt = cv2.imread(
            str(mask_PATH / m_p.name),
            cv2.IMREAD_GRAYSCALE,
        )
#         IoU = jaccard_score(gt, mask, average="micro")
        
        (
            total_small,
            detected_small,
            missed_small,
            wrong_small,
            overcounted_small,
            total_big,
            detected_big,
            missed_big,
            wrong_big,
            overcounted_big,
            n_rem
#         ) = missed_wrong_cysts(gt, mask, remove_thr=thresh)
        ) = missed_wrong_cysts(gt, mask, small_thresh=thresh)

        TPs_small += detected_small
        FPs_small += wrong_small
        FNs_small += missed_small

        TPs_big += detected_big
        FPs_big += wrong_big
        FNs_big += missed_big
        N_rem += n_rem
        
        ovb += overcounted_big
        ovs += overcounted_small
        
        # recall = detected / (detected + missed + 0.0001)
        # precision = detected / (detected + wrong + 0.0001)
        # precisions.append(precision)
        # recalls.append(recall)
#         import ipdb

#         ipdb.set_trace()

    TPss_list.append(TPs_small)
    FPss_list.append(FPs_small)
    FNss_list.append(FNs_small)
    TPsb_list.append(TPs_big)
    FPsb_list.append(FPs_big)
    FNsb_list.append(FNs_big)
    OVb.append(ovb)
    OVs.append(ovs)
    
    
    
TPss_list = np.array(TPss_list)
FPss_list = np.array(FPss_list)
FNss_list = np.array(FNss_list)
TPsb_list = np.array(TPsb_list)
FPsb_list = np.array(FPsb_list)
FNsb_list = np.array(FNsb_list)
    
    
IoUs = (TPss_list + TPsb_list) / (TPss_list + TPsb_list + FPss_list + FPsb_list + FNss_list + FNsb_list + EPSILON)
    # print(TPs, FPs, FNs)
precisions_small = (TPss_list / (TPss_list + FPss_list + EPSILON))
recalls_small = ((TPss_list / (TPss_list + FNss_list + EPSILON)))

precisions_big = ((TPsb_list / (TPsb_list + FPsb_list + EPSILON)))
recalls_big = ((TPsb_list / (TPsb_list + FNsb_list + EPSILON)))
    


metrics = [precisions_small, recalls_small, precisions_big, recalls_big, IoUs]
names = ["precision_small", "recall_small", "precision_big", "recall_big", "IoU"]
with open(out_PATH / "results_StoB.pickle", "wb") as outfile:
    pickle.dump([
        TPss_list, FPss_list, FNss_list, TPsb_list, FPsb_list, FNsb_list, OVb, OVs
    ], outfile)

fig, ax = plt.subplots(figsize=(12,8))

for i, m in enumerate(metrics):
    line_kwargs = {"drawstyle": "steps-post", "lw": 2, "c": colors[i]}
    ax.plot(thresholds, m, label=names[i], **line_kwargs)


plt.title(f"PR curve (on {len(list(pred_PATH.glob('*.png')))} test images)")
ax.legend(loc="lower left")
# plt.show()
plt.savefig(out_PATH / "PR_StoB_thr.png")
print("curve done")
print(f"--- Finished in {time() - t_start} ---")

# %%

# %%

# fig, ax = plt.subplots(2, 3, figsize=(25,10))
# for i, n_ex in enumerate([1,2,4]):
#     chard = len(list(Path(f"cv_perexp/exp{n_ex}/orig_names/").glob('*.png')))
#     with open(f"curves/exp_{n_ex}/results.pickle", 'rb') as file:
#         data = pickle.load(file)
#     TPss_list, FPss_list, FNss_list, TPsb_list, FPsb_list, FNsb_list, OVb, OVs = data
    
#     IoU = (TPss_list + TPsb_list) / (TPss_list + TPsb_list + FPss_list + FPsb_list + FNss_list + FNsb_list)
#     # print(TPs, FPs, FNs)
#     precision_small = (TPss_list / (TPss_list + FPss_list))
#     recall_small = ((TPss_list / (TPss_list + FNss_list)))

#     precision_big = ((TPsb_list / (TPsb_list + FPsb_list)))
#     recall_big = ((TPsb_list / (TPsb_list + FNsb_list)))
    
#     curves = [IoU, recall_small, precision_small, recall_big, precision_big]

#     thresholds = np.arange(0, 159, 3)
#     for d, l, c in zip(curves, lab2, colors2):
#         ax[0][i].plot(thresholds, d, label=l, c=c)#, **line_kwargs)
#         ax[0][i].legend()
#         ax[0][i].set_title(f'Exp {n_ex} [{chard} samples]')
#     for d, l, c in zip(data[:-2], labels[:-2], colors1):
#         ax[1][i].plot(thresholds, d, label=l, c=c)#, **line_kwargs)
#         ax[1][i].legend()
#     plt.tight_layout()
#     plt.savefig(f"curves/curves.png")
