import os
from IPython.display import clear_output
from queue import deque
from time import time

import json
import pickle
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageOps

from pathlib import Path, PureWindowsPath
# from training.utils_custom import cd
import pandas as pd
import seaborn as sns
from scipy.ndimage.morphology import binary_fill_holes
import re
from tqdm import tqdm
from scipy.sparse import csr_matrix


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
import argparse

rgbtohsv = RgbToHsv()
erosion = BinaryErosion(disk_size=5)
fill_holes = BinaryFillHoles()
remove_holes = RemoveSmallHoles(area_threshold=10000)
remove_objects = RemoveSmallObjects(min_size=5000)


ROOT = Path.cwd()
real_mask_PATH = ROOT / 'artifacts/dataset:v10/masks'
real_annot_PATH = ROOT / 'annotations_original/full_dataset'
real_img_PATH = ROOT / 'artifacts/dataset:v10/images'
# real_annot_PATH = ROOT / 'annotations_original/all_oldnames'
# real_img_PATH = ROOT / 'images_original/all_oldnames/'

def get_args():
    parser = argparse.ArgumentParser(description='CV with selected experiment as test set and train/val stratified from the others')
    parser.add_argument("-i", "--inpath", type=Path, help="Path containing image predicitions.", required=True)
    parser.add_argument("-d", "--dataset", default=None)
    
    return parser.parse_args()


# 3 µm (valore minimo) ad uno di circa 40 µm (valore massimo). I valori medi invece sono tra i 6 e i 13 µm
# 25 : 72 = µm : px
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
    return [cX, cY]

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
    mask = np.stack((~mask, mask, np.zeros_like(mask))).T.astype(np.uint8)
    
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

from simplify_names import get_packs

def unpack_name(name):
    try:
        date, treatment, tube, zstack, side = get_packs(name.strip())
    except:
        print(f'Error for {name}')
        assert False
        
    return {
        'date' : date,  # pd.Timestamp(tmp[1].split('-')[-1]),
        'treatment' : treatment,
        'tube' : tube,
        'zstack' : zstack,
        'side' : side if side != 'U' else '-'
    }

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
    date_exps = {'0919': 1, '1019': 2, '072020':3, '092020':3, '122020':4, '0721':5}
    date = ''.join((date).split('.')[1:])
    return date_exps[date]
# def date_to_exp(date):
#     if type(date) == pd.Timestamp:
#         date = f"{date.day}{date.year}" # actually this is "month.year"
#     else:
#         date = "".join(date.split('.')[1:])
    
#     date_exps = {'0919': 1, '1019': 2, '072020':3, '092020':3, '122020':4, '0721':5}
#     return date_exps[date]
    
def mean_density(df):
    means = np.array([df[(df.date==d) & (df.treatment=='CTRL')].density.mean() for d in df.date.unique() if 'CTRL' in df[df.date==d].treatment.unique()])
    idx = (np.abs(means - means.mean())).argmin()
    return means[idx]

def showlinks(df):
    domain = 'https://jupyter.polito.it/expert/hub/user-redirect/lab/tree/my_rene-policistico/results/'
    df.index = df.index.to_series().apply(lambda x: f'<a href="{domain}unet_tiles_{n_tiles}/{quote(x)}.jpg?redirects=0">{x}</a>')
    df = df.to_html(render_links=True, escape=False)
    display(HTML(df))


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

def show(name, n_ex=None, res_path=None, title=None):
    fig, ax = plt.subplots(1,3,figsize=(20,6))
    if n_ex:
        res_model_PATH = Path('.') / f'cv_perexp/exp{n_ex}/orig_names'
    else:
        res_model_PATH = res_path
    
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



def missed_wrong_cysts_dict(gt: np.array, pred: np.array, cutoff=288): #TODO: Specify returns
    cysts = []
    
    gt_contours, _ = cv2.findContours(gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    gt_contours = np.array([c for c in gt_contours if c.size > 4 and cv2.contourArea(c)>cutoff], dtype=object)
    pred_contours = np.array([c for c in pred_contours if c.size > 4 and cv2.contourArea(c)>cutoff], dtype=object)
    
    gt_seps = np.array([csr_matrix(cv2.fillPoly(np.zeros_like(gt), pts=[c], color=(1))) for c in gt_contours], dtype=object)
    
    pred_seps = np.array([csr_matrix(cv2.fillPoly(np.zeros_like(gt), pts=[c], color=(1))) for c in pred_contours], dtype=object)

    detect_miss_list = np.array([False for _ in pred_seps])

    count = (i for i in range(len(gt_contours) + len(pred_contours)))
    

    for single_gt, c in zip(gt_seps, gt_contours):
        curr_detection = np.array([single_gt.multiply(sing_p).count_nonzero() for sing_p in pred_seps], dtype=bool)

        if curr_detection.any():
            best_p = np.argmax([cv2.contourArea(cp) for cp in pred_contours[curr_detection]])
            areas = (cv2.contourArea(c), cv2.contourArea(pred_contours[curr_detection][best_p]))
#            centers = (center(c), center(pred_contours[curr_detection][best_p]))
            cysts.append({'state': 'detected', 
    					'area_real': areas[0], 
    					'area_pred': areas[1], 
#             					'centers': centers
    					})
            for c_p in np.hstack((pred_contours[curr_detection][:best_p], pred_contours[curr_detection][best_p+1:])):
                areas = (cv2.contourArea(c), cv2.contourArea(c_p))
#                centers = (center(c), center(pred_contours[curr_detection][best_p]))
                cysts.append({'state': 'overcounted',
        					'area_real': areas[0], 
        					'area_pred': areas[1], 
#             					'centers': centers
        					})

        else:
            areas = (cv2.contourArea(c), None)
#            centers = (center(c), None)
            cysts.append({'state': 'missed',
    					'area_real': areas[0], 
    					'area_pred': areas[1],
#            					'centers': centers
    					})

    sparse_gt = csr_matrix(gt)
    for single_pred, c in zip(pred_seps, pred_contours):
        if not single_pred.multiply(sparse_gt).count_nonzero():
            areas = (None, cv2.contourArea(c))
#            centers = (None, center(c))
            cysts.append({'state': 'wrong',
    					'area_real': areas[0], 
    					'area_pred': areas[1], 
#            					'centers': centers
    					})

    return cysts, len(gt_contours), len(pred_contours)


def write_results(folder:Path, is_jpg=False, maskp=None, imgp=None):
    folder.mkdir(exist_ok=True)
    np.seterr('raise')
    print("WRITE RESULTS", folder)
    datafile_im = folder / "images_table.csv"
    datafile_cy = folder / "cysts_table.csv"
    if datafile_im.exists():
        print(f"> Table in {folder.stem} already exists!")
        return
    IM_dict = {}
    
#     stack = deque([{}, {}])
    
#     suffix = '*.jpg' if is_jpg else '*.png'

    IM_df = pd.DataFrame([])
    CYST_df = pd.DataFrame([])
    
    eval_name = folder.stem
    
    paths = sorted(folder.glob('*.png'))
    print(len(paths))
#     paths = random.sample(paths, 30)
    for i, pred in enumerate(tqdm(paths, desc=str(folder))):
        name = pred.stem
        IM_s = pd.Series({"Analysis": eval_name}, name=name) 
        CYST_s = pd.Series({"Analysis": eval_name}) # Cyst row with {'state': state, AREA_real, AREA_pred, 'centers': [(x_r, y_r), (x_p, y_p)]}
        
        CYST_s["name"] = name
        for s, v in unpack_name(name).items():
            IM_s[s] = v
            CYST_s[s] = v

        # dict of cysts as {'state': state, 'areas': [AREA_real, AREA_pred], 'centers': [(x_r, y_r), (x_p, y_p)]}
#         s.name = get_old_name(name)            
            
        assert (maskp / f'{name}.png').exists(), maskp / f'{name}.png'
        gt = open_mask(maskp / f'{name}.png')
        pred_img = open_mask(pred)
        
        cysts, IM_s['total_real'], IM_s['total_pred'] = missed_wrong_cysts_dict(gt, pred_img, cutoff=0)
        
        IM_s['tube_area'] = tube_area(name)
            
        # s['#recall'] = s['detected']/(s['detected'] + s['missed'] + 0.0001)
        
        gt = gt.ravel()
        # gt = np.minimum(gt, np.ones_like(gt))
        pred_img = pred_img.ravel()
        # pred_img = np.minimum(pred_img, np.ones_like(pred_img))
        
        cf = confusion_matrix(gt, pred_img).ravel() if gt.any() else [0, 0, 0, 0]
        TN, FP, FN, TP = cf #if len(cf)==4 else [0, 0, 0, 0]
        IM_s['pxTP'] = int(TP)
        IM_s['pxFN'] = int(FN)
        IM_s['pxFP'] = int(FP)
        IM_s['pxTN'] = int(TN)
        
        IM_s['iou'] = float(TP / (TP + FN + FP + .001))
        IM_s['recall'] = float(TP / (TP + FN + .001))
        IM_s['precision'] = float(TP / (TP + FP + .001))
        IM_s['mcc'] = float(mcc(TP, TN, FP, FN))

        IM_df = IM_df.append(IM_s)
        # TODO: Test if it works
        for c in cysts:
            CYST_df = CYST_df.append(CYST_s.append(pd.Series(c)),
                                    ignore_index=True)
        
#     json.dump(IM_dict, open(datafile, 'w'))
    IM_df.to_csv(datafile_im)
    CYST_df.to_csv(datafile_cy)
    print(f'Results saved in "{datafile_im.parent}"')
    return
    
def read_from_splits(path):
    with open(path, "rb") as f:
        sp = pickle.load(f)
        
    samples = []
    for key, s in sp.items():
        for l in s:
            tmp = unpack_name(l)
            unpaks = [key] + list(tmp.values())
            unpaks.append(date_to_exp_n(unpaks[1]))
            samples.append(unpaks)

    return pd.DataFrame(samples, columns=['key']+list(tmp.keys())+['exp'])
    

if __name__ == '__main__':
    args = get_args()
    res_model_PATH = args.inpath
    
    if args.dataset is not None:
        d_fold = f'artifacts/dataset:{args.dataset}'
    else:
        d_fold = sorted(Path('artifacts').iterdir(), key=lambda n: int(n.stem.split(':v')[-1]))[-1]
        
    real_mask_PATH = d_fold / 'masks'
    real_img_PATH = d_fold / 'images'
    
    t0 = time()
    write_results(res_model_PATH, is_jpg=False, maskp=real_mask_PATH, imgp=real_img_PATH)
    print(f"\n__________________ Finished in {time()-t0} s __________________")
