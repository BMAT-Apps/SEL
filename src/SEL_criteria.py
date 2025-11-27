#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 16:34:28 2025

@author: colin
"""

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, HuberRegressor
from skimage.morphology import binary_erosion
from scipy.spatial import distance
from scipy.ndimage import center_of_mass


def concentricity(mask, jacobian):
    
    lesions = list(np.unique(mask))
    if 0 in lesions:
        lesions.remove(0)
    lesions_concentricity = pd.DataFrame(columns=['label_id', 'concentricity'])
    i = 0
    for les in lesions:
        lesions_concentricity.at[i, 'label_id'] = les
        les_mask = np.zeros_like(mask)
        les_mask[mask == les] = 1
        les_jac = jacobian.copy()
        les_jac[mask != les] = 0
        les_concentricity = concentricity_com(les_mask, les_jac)
        lesions_concentricity.at[i, 'concentricity'] = les_concentricity
        i = i+1
    
    return lesions_concentricity


def concentricity_les(mask, jacobian, min_vol=10):
    
    # footprint = np.array([[[False, False, False],
    #         [False,  False, False],
    #         [False, False, False]],

    #        [[False,  True, False],
    #         [ True, False,  True],
    #         [False,  True, False]],

    #        [[False, False, False],
    #         [False,  False, False],
    #         [False, False, False]]])
        
    circles_exp = []
    
    w = np.where(mask == 1)
    it = 0
    while len(w[0]) > 0 and it < 100:
        mask_e = binary_erosion(mask).astype(np.float32)
        c = np.where((mask-mask_e) == 1)
        c_exp = np.mean(jacobian[c])
        circles_exp.append(c_exp)
        mask = mask_e
        w = np.where(mask == 1)
        it += 1
        
    if len(circles_exp) < 2:
        return "N/A"
        
    # compute concentricy
    x = np.array(range(len(circles_exp))).reshape(-1, 1)
    y = np.array(circles_exp).reshape(-1, 1)
    
    LR = HuberRegressor().fit(x, y)
    
    return LR.coef_[0].item()


def concentricity_les_bis(mask, jacobian, min_vol=10):
    
    lesions_w = np.where(mask == 1)
    lesions_jac = jacobian[lesions_w]
    
    mask_com = weighted_com(lesions_w, [1]*len(lesions_w[0]))
    jac_com = weighted_com(lesions_w, lesions_jac)
    
    return -distance.euclidean(mask_com, jac_com)

def concentricity_com(mask, jacobian, min_vol=10):
    
    mask_bands = com_bands(mask)
    
    band_u = list(np.unique(mask_bands))
    band_u.remove(0)
    band_u = sorted(band_u, reverse=True)
    
    bands_jac = []
    for b in band_u:
        w = np.where(mask_bands == b)
        b_jac = np.mean(jacobian[w])
        bands_jac.append(b_jac)
    
    if len(bands_jac) < 2:
        return "N/A"
    
    # compute concentricy
    x = np.array(range(len(bands_jac))).reshape(-1, 1)
    y = np.array(bands_jac).reshape(-1, 1)
    
    LR = HuberRegressor().fit(x, y)
    
    return LR.coef_[0].item()


def com_bands(mask):
    # mask: 3D boolean array
    com = center_of_mass(mask)  # (z, y, x)

    # coordinate grid
    zz, yy, xx = np.indices(mask.shape)

    # Euclidean distance from centroid
    dist = np.sqrt((zz - com[0])**2 + (yy - com[1])**2 + (xx - com[2])**2)
    
    # Only consider voxels inside the mask
    dist_in_mask = dist * mask
    
    # Assign band index by rounding or flooring
    bands = np.ceil(dist_in_mask).astype(int)
    
    return bands
    


def weighted_com(lesion_where, lesion_weights):
    
    total_weight = np.sum(lesion_weights)
    if total_weight == 0:
        return None  # No valid CoM

    # Compute weighted CoM
    x_com = np.sum(lesion_weights * lesion_where[0]) / total_weight
    y_com = np.sum(lesion_weights * lesion_where[1]) / total_weight
    z_com = np.sum(lesion_weights * lesion_where[2]) / total_weight

    return (x_com, y_com, z_com)


def constancy(masks, jacobians):
    
    if not (type(masks) == list and type(jacobians) == list):
        print("masks and jacobians should be list")
        return
        
    if len(masks) != len(jacobians):
        print("masks should be the same size of jacobians")
        return
    
    lesions_u = []
    for mask in masks:
        mask_u = list(np.unique(mask))
        lesions_u.extend(mask_u)
    lesions = list(np.unique(lesions_u))
    lesions.remove(0)
    if 0 in lesions:
        lesions.remove(0)
    lesions_constancy = pd.DataFrame(columns=['label_id', 'constancy', 'expansion_slope'])
    i = 0
    for les in lesions:
        
        lesions_constancy.at[i, 'label_id'] = les
        
        les_jacs = []
        for mask, jac in zip(masks, jacobians):
            les_w = np.where(mask == les)
            if len(les_w[0]) == 0:
                les_jacs = None
                break
            
            mean_exp = np.mean(jac[les_w])
            les_jacs.append(mean_exp)
            
        if les_jacs:
            les_constancy, les_exp = constancy_les(les_jacs)
            lesions_constancy.at[i, 'constancy'] = les_constancy
            lesions_constancy.at[i, 'expansion_slope'] = les_exp
        i = i+1
    
    return lesions_constancy
            
            
def constancy_les(les_jacs):
    les_jacs.insert(0, 0)
    x = np.array(range(len(les_jacs))).reshape(-1, 1)
    les_jacs = np.array(les_jacs)*x.reshape(-1)
    y = les_jacs.reshape(-1, 1)
    
    LR = LinearRegression(fit_intercept=False).fit(x, y)
        
    errors = []
    for i in range(len(les_jacs)):
        if i == 0:
            continue     
        pi = LR.predict(np.array(i).reshape(-1, 1)).item()
        ei = np.power(((les_jacs[i] - pi)/pi), 2)
        errors.append(ei)
    
    return np.mean(errors), LR.coef_.item()