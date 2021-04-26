import os
from IPython.display import clear_output
from queue import deque

import json
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Polygon
from PIL import Image

from PIL import Image
import PIL.ImageDraw as ImageDraw
from PIL import ImageOps
from pathlib import Path, PureWindowsPath
# from training.utils_custom import cd
import pandas as pd
import seaborn as sns
from scipy.ndimage.morphology import binary_fill_holes
import re
from tqdm import tqdm

from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, confusion_matrix, matthews_corrcoef
from IPython.core.display import display, HTML
from urllib.parse import quote


from histolab.filters.morphological_filters import (
    BinaryFillHoles,
    BinaryErosion,
    RemoveSmallHoles,
    RemoveSmallObjects,   
)
from histolab.filters.image_filters import RgbToHsv

from scipy.ndimage.morphology import binary_fill_holes, binary_opening
from skimage import filters
from skimage.filters import try_all_threshold
from skimage import exposure

rgbtohsv = RgbToHsv()
erosion = BinaryErosion(disk_size=5)
fill_holes = BinaryFillHoles()
remove_holes = RemoveSmallHoles(area_threshold=10000)
remove_objects = RemoveSmallObjects(min_size=5000)


ROOT = Path('DATASET')
real_mask_PATH = ROOT / 'masks/all_oldnames'
real_annot_PATH = ROOT / 'annotations_original/all_oldnames'
real_img_PATH = ROOT / 'images_original/all_oldnames/'
res_model_PATH = ROOT / 'results/seed_0_cv'


# 3 µm (valore minimo) ad uno di circa 40 µm (valore massimo). I valori medi invece sono tra i 6 e i 13 µm
#     25 : 72 = µm : px
to_px = lambda m : m * 72 / 25
to_m = lambda px : px * 25 / 72
to_area_m = lambda Apx : (25/72)**2 * Apx
area_mc = lambda Apx : (25/72)**2 * Apx * 1e-6 ## to mm²

circ_area = lambda d : np.pi * (d/2)**2

min_area = circ_area(3)
max_area = circ_area(40)
mid_areas = circ_area(np.array([6, 13]))

## FUNCTIONS

def open_mask(path):
    im = str(path)
    return cv2.imread(im, cv2.IMREAD_GRAYSCALE)

def center(c):
    M = cv2.moments(c)
    if M["m00"] == 0: print(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return np.array([cX, cY])

# def counter_withmemory(stack, min_dist=30):
#     name = next(iter(stack[0].keys()))
#     current_centers = stack[0][name]
# #     n = len(current_centers)
#     n = 0
#     tmp = name.split('40x')
#     z = tmp[1].split('_')
#     for prev_centers in (stack[1], stack[2]):
#         prev_centers = prev_centers.get(f"{tmp[0]}40x{(int(z[0])-1):02}_{z[1]}")
#         if prev_centers:
#             for c in current_centers:
#                 if (np.array([np.linalg.norm(c-cp) for cp in prev_centers]) < min_dist).any():
#                     n -= 1
#     return n

def counter_withmemory(centers_list, state_list, debug=False, min_dist=60):
    focus = ["missed", "detected"] if debug else ["wrong", "detected"]
    centers_list = [np.array(centers_list[i])[[s in focus for s in state_list[i]]] for i in range(len(centers_list))]
    n = len(centers_list[0])
    
    for idx in range(1, len(centers_list)):
        for c in centers_list[idx]:
            if not (np.array([np.linalg.norm(np.array(c)-np.array(cp)) for cp in centers_list[idx-1]]) < min_dist).any():
                    n += 1
    return n

def tube_area(fname: str):
    fname = real_img_PATH / f"{fname}.jpg"
    rgb_img = Image.open(fname)
    
    with np.errstate(all='ignore'):
        hsv_img = rgbtohsv(rgb_img)
        last = np.array(hsv_img)[:,:,2]
        sobel = filters.sobel(last)
        blurred = filters.gaussian(sobel, sigma=5.0)
        blurred = exposure.adjust_sigmoid(blurred)
    
    binary_image = blurred > (filters.threshold_otsu(blurred)*.67)
    
    image_remove_holes = remove_holes(binary_image)
    image_remove_holes = binary_opening(image_remove_holes, structure=np.ones((20,20)))
    
    image_filled = remove_objects(image_remove_holes)
    image_remove_object = fill_holes(image_filled)
    mask = erosion(image_remove_object)
    return area_mc(mask.sum())

def show_tube(fname, title=None):
    fname = real_img_PATH / f"{fname}.jpg"
    rgb_img = Image.open(fname)
    
    with np.errstate(all='ignore'):
        hsv_img = rgbtohsv(rgb_img)
        last = np.array(hsv_img)[:,:,2]
        sobel = filters.sobel(last)
        blurred = filters.gaussian(sobel, sigma=5.0)
        blurred = exposure.adjust_sigmoid(blurred)
    
    binary_image = blurred > (filters.threshold_otsu(blurred)*.67)
    
    image_remove_holes = remove_holes(binary_image)
    image_remove_holes = binary_opening(image_remove_holes, structure=np.ones((20,20)))
    
    image_filled = remove_objects(image_remove_holes)
    image_remove_object = fill_holes(image_filled)
    mask = erosion(image_remove_object)
    mask = np.stack((mask, np.zeros_like(mask), np.zeros_like(mask))).T.astype(np.uint8)
    
    mask_img = ImageOps.flip(Image.fromarray(mask*255).rotate(90))
    mask_img.putalpha(60)
    rgb_img.paste(im=mask_img, box=(0, 0), mask=mask_img)
    
    if title: plt.title(title)
    plt.imshow(rgb_img)
    plt.axis('off')
    plt.show()
    

# show_tube("experiment_29-30.07.2020_CTRL_1_40x04_A")


def calc_full_sizes(mask, name, thr=10):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, 
                                    cv2.CHAIN_APPROX_SIMPLE)
    real = open_mask(real_mask_PATH / f"{name}.png")
    
    state=[]
    contours = [c for c in contours if cv2.contourArea(c) > thr]
    areas = [cv2.contourArea(c) for c in contours]
    n = len(contours)
    
    for a,c in zip(areas, contours):
        x,y = center(c)
        diam = np.sqrt(4*a/np.pi)
        if (mask.shape[0] - x) < diam/2 or (mask.shape[1] - y) < diam/2 : n -= 1#; print(name)
            
        single_c = np.zeros_like(mask)
        cv2.fillPoly(single_c, pts=[c], color=(1))
        if np.logical_and(single_c, real).any():
            state.append('detected')
        else:
            state.append('wrong')

    return areas, state

from simplify_names import unpack_name as get_packs

def unpack_name(name):
    date, treatment, tube, zstack, side = get_packs(name.strip())
    dc = {
        'date' : date,  # pd.Timestamp(tmp[1].split('-')[-1]),
        'treatment' : treatment,
        'tube' : tube,
        'zstack' : zstack,
        'side' : side if side != 'U' else '-'
    }
    return dc

def exp_to_dates(n):
    '''
    exp 1: sett 2019; exp2: ott 2019; exp3: lug e sett 2020; exp4: dicembre 2020
    '''
    if n == 0: return ['4-12.09.19', '9-17.10.19', '29-30.07.2020', '2-3.09.2020', '11.12.2020', '18.12.2020']
    if n == 1: return ['4-12.09.19']
    if n == 2: return ['9-17.10.19']
    if n == 3: return ['29-30.07.2020', '2-3.09.2020']
    if n == 4: return ['11.12.2020', '18.12.2020']
    
def date_to_exp(date):
    if type(date) == pd.Timestamp:
        date = f"{date.day}{date.year}" # actually this is "month.year"
    else:
        date = "".join(date.split('.')[1:])
    
    date_exps = {'0919': 1, '1019': 2, '072020':3, '092020':3, '122020':4}
    return date_exps[date]
    
def mean_density(df):
    means = np.array([df[(df.date==d) & (df.treatment=='CTRL')].density.mean() for d in df.date.unique() if 'CTRL' in df[df.date==d].treatment.unique()])
    idx = (np.abs(means - means.mean())).argmin()
    return means[idx]

def showlinks(df):
    domain = 'https://jupyter.polito.it/expert/hub/user-redirect/lab/tree/my_rene-policistico/results/'
    df.index = df.index.to_series().apply(lambda x: f'<a href="{domain}unet_tiles_{n_tiles}/{quote(x)}.jpg?redirects=0">{x}</a>')
    df = df.to_html(render_links=True, escape=False)
    display(HTML(df))

# def missed_wrong_cysts_finer(gt, pred, thr=30, min_overlap=.3):
#     gt_contours, _ = cv2.findContours(gt, cv2.RETR_TREE, 
#                                     cv2.CHAIN_APPROX_SIMPLE) 
#     pred_contours, _ = cv2.findContours(pred, cv2.RETR_TREE, 
#                                     cv2.CHAIN_APPROX_SIMPLE)
    
#     detected = 0
#     for i, c in enumerate(gt_contours):
#         single_gt = np.zeros_like(gt)
#         cv2.fillPoly(single_gt, pts=[c], color=(1))
        
#         for j, cp in enumerate(pred_contours):
#             if cv2.contourArea(cp) < thr: continue            # thr ~ 1/2 theoretical minimum
#             single_pred = np.zeros_like(pred)
#             cv2.fillPoly(single_pred, pts=[cp], color=(1))
            
#             I = np.logical_and(single_gt, single_pred)
#             overlap = I.sum()/single_gt.sum()
#             if overlap  > min_overlap:
#                 detected +=1
    
#     missed = len(gt_contours) - detected
#     wrong = len(pred_contours) - detected

#     return len(gt_contours), detected, missed, wrong

def overlapping_areas(gt, pred):
    gt_contours, _ = cv2.findContours(gt, cv2.RETR_TREE, 
                                    cv2.CHAIN_APPROX_SIMPLE) 
    pred_contours, _ = cv2.findContours(pred, cv2.RETR_TREE, 
                                    cv2.CHAIN_APPROX_SIMPLE)
    
    percs = []
    
    for i, c in enumerate(gt_contours):
        single_gt = np.zeros_like(gt)
        cv2.fillPoly(single_gt, pts=[c], color=(1))
        
        for j, cp in enumerate(pred_contours):
#             print(f"testing {i}-{j} overlap")
            single_pred = np.zeros_like(pred)
            cv2.fillPoly(single_pred, pts=[cp], color=(1))
            I = np.logical_and(single_gt, single_pred)
            if I.any():
                percs.append(single_pred.sum()/single_gt.sum())
#                 percs.append(I.sum()/single_gt.sum())
        
    return percs

def get_old_name(name):
    path = real_annot_PATH / f"{name}.json"
    if not path.exists(): return name
    with path.open('r') as data:
        real_annot = json.load(data)
        rname = real_annot["imagePath"]
    return PureWindowsPath(rname).stem

def get_new_name(date, treatment, tube, zstack, side):
    if side=='-':
        side = 'U'
        
    return f'experiment_{date}_{treatment}_{tube}_40x{zstack:02}_{side}'

def show(name, n_ex, title=None):
    fig, ax = plt.subplots(1,3,figsize=(20,6))
    res_model_PATH = ROOT / f'cv_perexp/exp{n_ex}'
    
    img = Image.open(real_img_PATH / f"{name}.jpg")
    mask = Image.open(real_mask_PATH / f"{name}.png")
    pred = Image.open(res_model_PATH / f"{name}.png")
#     pred = open_mask(res_model_PATH / f"{name}.png")
    
    if title: fig.suptitle(title)
    for a, im, tit in zip(ax, [img, mask, pred], ['img', 'mask', 'pred']):
        a.axis('off')
        a.set_title(tit)
        a.imshow(im)
    plt.show()

def clean_mask(img, thr = 58):
    blank = np.zeros_like(img)
    contours,_ = cv2.findContours(img, cv2.RETR_TREE, 
                                    cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area > thr:
            cv2.fillPoly(blank, pts =[c], color=(255))
    
    return blank

def mcc(TP, TN, FP, FN, N=None):
    if N==None:
        N = TN + TP + FN + FP
    if N == 0: return 1
    S = (TP + FN)/N
    P = (TP + FP)/N
    
    return (TP/N - S*P) / np.sqrt(0.001 + P*S*(1-S)*(1-P))



# from PIL import Image
# from pathlib import Path
# import cv2
# np.seterr(divide='ignore', invalid='ignore')

# from histolab.filters.morphological_filters import (
#     BinaryDilation,
#     BinaryFillHoles,
#     BinaryErosion,
#     RemoveSmallHoles,
#     RemoveSmallObjects,
    
# )
# from histolab.filters.image_filters import (
#     RgbToGrayscale,
#     OtsuThreshold,
#     ToPILImage,
#     LocalOtsuThreshold,
#     RgbToHsv,
    
# )
# from scipy.ndimage.morphology import binary_fill_holes, binary_opening
# from skimage import filters
# from skimage.filters import try_all_threshold
# from skimage import exposure

# import random
# import os
# import matplotlib.pyplot as plt
# import numpy as np
# from time import time


# def old_pipeline(path: str):
#     img = cv2.imread(str(path))
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray,(5,5),0)
#     _, thresh = cv2.threshold(gray,220,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
#     im_floodfill = thresh.copy()

#     h, w = thresh.shape[:2]
#     mask = np.zeros((h+2, w+2), np.uint8)
    
#     _,_, mask, _ = cv2.floodFill(im_floodfill, mask, (0,0), 255);
    
#     maskv = np.vstack([np.ones(thresh.shape[1]), thresh, np.ones(thresh.shape[1])])
#     maskv = binary_fill_holes(maskv)[1:-1, :]
    
#     maskh = np.column_stack([np.ones(thresh.shape[0]), thresh, np.ones(thresh.shape[0])])
#     maskh = binary_fill_holes(maskh)[:, 1:-1]
    
#     mask = np.maximum(maskh, maskv)
#     return mask

# PATH = "images_original/full_dataset/"

# rgbtohsv = RgbToHsv()
# erosion = BinaryErosion(disk_size=5)
# fill_holes = BinaryFillHoles()
# remove_holes = RemoveSmallHoles(area_threshold=10000)
# remove_objects = RemoveSmallObjects(min_size=5000)

# def detect_tissue_debug(fname: str, debug=False):
# #     fname = os.listdir(path)[i]
# #     rgb_img = Image.open(os.path.join(path, fname))
#     fname = Path(PATH) / fname
#     rgb_img = Image.open(fname)
#     if debug:
#         plt.imshow(rgb_img)
#         plt.title('real image')
#         plt.show()
    
#     hsv_img = rgbtohsv(rgb_img)
#     last = np.array(hsv_img)[:,:,2]
#     sobel = filters.sobel(last)
#     blurred = filters.gaussian(sobel, sigma=5.0)
# #     blurred = exposure.rescale_intensity(blurred)
#     blurred = exposure.adjust_sigmoid(blurred)
#     if debug:
#         plt.imshow(blurred)
#         plt.title('grayscale')
#         plt.show()
    
#     binary_image = blurred > (filters.threshold_otsu(blurred)*.67)
#     if debug:
#         plt.imshow(binary_image)
#         plt.title('thresholded')
#         plt.show()
    
#     image_remove_holes = remove_holes(binary_image)
#     if debug:
#         plt.imshow(image_remove_holes)
#         plt.title('remove holes')
#         plt.show()
    
#     image_remove_holes = binary_opening(image_remove_holes, structure=np.ones((20,20)))
#     if debug:
#         plt.imshow(image_remove_holes)
#         plt.title('opening')
#         plt.show()
    
#     image_filled = remove_objects(image_remove_holes)
#     image_remove_object = fill_holes(image_filled)
#     image = erosion(image_remove_object)
#     return rgb_img, ToPILImage()(image)

# def test_area(i):
#     name = f"experiment_29-30.07.2020_T4_2_40x{i}_A.jpg"
#     file = Path(PATH) / name
#     if not file.exists(): return None
    
#     img, mask = detect_tissue(name)
#     old_mask = old_pipeline(file)
#     fig,ax = plt.subplots(1, 2, figsize=(14,5))
#     fig.suptitle(f'image n°{i}')
#     ax[0].imshow(img)
    
#     mask = np.array([np.asarray(mask), 255*old_mask, 255*old_mask]).transpose(1, 2, 0)

#     ax[1].imshow(mask)
    
#     plt.show()

# t0= time()
# for i in range(49):
#     test_area(i)
#     if i > 20: break
# print(time()-t0)