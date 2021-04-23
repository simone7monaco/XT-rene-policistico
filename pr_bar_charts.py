# from basic_units import cm, inch
#%%
import re

import matplotlib.pyplot as plt
import numpy as np

thresholds = []
precisions_big = []
recalls_big = []
precisions_small = []
recalls_small = []
IoUs = []

log_file = "pr_curve_small_first.log"

with open(log_file) as f:
    f = f.readlines()

for line in f:
    if "thresh" in line:
        thresholds.append(line)
    elif "precision big" in line:
        precisions_big.append(line)
    elif "recall big" in line:
        recalls_big.append(line)
    elif "precision small" in line:
        precisions_small.append(line)
    elif "recall small" in line:
        recalls_small.append(line)
    elif "IoU" in line:
        IoUs.append(line)


thresholds = [
    round(float(re.findall("\d+\.\d+", thresholds[i])[0]), 3)
    for i in range(len(thresholds))
]  # [9:12]

recalls_big = [
    float(re.findall("\d+\.\d+", recalls_big[i])[0]) for i in range(len(recalls_big))
]  # [9:12]
precisions_big = [
    float(re.findall("\d+\.\d+", precisions_big[i])[0])
    for i in range(len(precisions_big))
]  # [9:12]

recalls_small = [
    float(re.findall("\d+\.\d+", recalls_small[i])[0])
    for i in range(len(recalls_small))
]  # 9:12]

precisions_small = [
    float(re.findall("\d+\.\d+", precisions_small[i])[0])
    for i in range(len(precisions_small))
]  # [9:12]

IoUs_mean = [
    float(re.findall("\d+\.\d+", IoUs[i])[0]) for i in range(len(IoUs))
]  # [9:12]
IoUs_std = [
    float(re.findall("\d+\.\d+", IoUs[i])[1]) for i in range(len(IoUs))
]  # [9:12]

N = len(thresholds)


#%%
width = 0.5
fig, ax = plt.subplots(figsize=(10, 10))


ind = 4 * np.arange(N)

ax.bar(ind, recalls_big, width, bottom=0, label="Recall big")

ax.bar(ind + width, precisions_big, width, bottom=0, label="Precision big")


ax.bar(
    2 * width,
    [0.2, 0.6, 0.1],
    width,
    bottom=0,
    label="Precision small",
)

ax.bar(
    ind + 3 * width,
    recalls_small,
    width,
    label="Recall small",
)


ax.bar(
    ind + 4 * width,
    IoUs_mean,
    width,
    bottom=0,
    yerr=IoUs_std,
    label="IoU mean",
)


ax.set_title("Metrics by threshold")
ax.set_xticks(ind + 3.5 * width / 2)
ax.set_xticklabels(thresholds)
plt.xticks(rotation=90)

ax.legend()

plt.show()

# %%
