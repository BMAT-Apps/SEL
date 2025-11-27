#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 14:29:09 2025

@author: colin
"""
import os
from os.path import join as pjoin
from os.path import exists as pexists
import sys
import subprocess
import shutil
import argparse
import nibabel as nib
import numpy as np
import pandas as pd
from bids_lesion_expansion import bids_lesion_expansion
from bids_SEL_candidates import bids_SEL_candidates
from SEL_criteria import concentricity, constancy


def bids_SEL(bids, sub, ses01, ses02, ses03, deriv='SEL', flair='FLAIR', mprage='acq-MPRAGE_T1w', deriv_mask='nnUNet', mask_name='mask-bin_algo-nnUNet_FLAIR', samseg=False):
    
    print('Checking files...')
    # check files
    for ses in [ses01, ses02, ses03]:
        if not pexists(pjoin(bids, f'sub-{sub}', f'ses-{ses}', 'anat', f'sub-{sub}_ses-{ses}_{flair}.nii.gz')):
            print(f'[ERROR] no ses {ses} FLAIR image')
            return 
        
        if not pexists(pjoin(bids, f'sub-{sub}', f'ses-{ses}', 'anat', f'sub-{sub}_ses-{ses}_{mprage}.nii.gz')):
            print(f'[ERROR] no ses {ses} MPRAGE image')
            return 
        
        if not samseg:
            if not pexists(pjoin(bids, 'derivatives', deriv_mask, f'sub-{sub}', f'ses-{ses}', f'sub-{sub}_ses-{ses}_{mask_name}.nii.gz')):
                print(f'[ERROR] no ses {ses} binary lesion mask')
                return
    
    # Compute espansion between the 3 sessionss
    print('Compute expansion between sessions...')
    if not pexists(pjoin(bids, 'derivatives', deriv, f'ses-{ses01}-{ses02}', f'sub-{sub}_ses-{ses01}_to_ses-{ses02}_jacobian.nii.gz')):
        bids_lesion_expansion(bids, sub, ses01, ses02, mprage=mprage, flair=flair, deriv=deriv)
    if not pexists(pjoin(bids, 'derivatives', deriv, f'ses-{ses01}-{ses03}', f'sub-{sub}_ses-{ses01}_to_ses-{ses03}_jacobian.nii.gz')):
        bids_lesion_expansion(bids, sub, ses01, ses03, mprage=mprage, flair=flair, deriv=deriv)
    
    # Register each binary lesion mask in the T1w space
    if samseg:
        print('Compute binary lesion mask with SAMSEG...')
        mask_prob = 'mask-prob_algo-SAMSEG_T1w'
        mask_t1 = 'mask-bin_algo-SAMSEG_T1w'
        for ses in [ses01, ses02, ses03]:
            sub_ses_deriv = pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses}')
            if not pexists(pjoin(sub_ses_deriv, f'sub-{sub}_ses-{ses}_{mask_prob}.nii.gz')):
                subprocess.Popen(f"run_samseg -i {pjoin(sub_ses_deriv, f'sub-{sub}_ses-{ses}_MPRAGE.nii.gz')} -i {pjoin(sub_ses_deriv, f'sub-{sub}_ses-{ses}_FLAIR.nii.gz')} --lesion --lesion-mask-pattern 0 1 --threads 8 -o {pjoin(sub_ses_deriv, 'SAMSEG')} --save-posteriors", shell=True).wait()
                subprocess.Popen(f"mri_convert {pjoin(sub_ses_deriv, 'SAMSEG', 'posteriors', 'Lesions.mgz')} {pjoin(sub_ses_deriv, f'sub-{sub}_ses-{ses}_{mask_prob}.nii.gz')}", shell=True).wait()
            if not pexists(pjoin(sub_ses_deriv, f'sub-{sub}_ses-{ses}_{mask_t1}.nii.gz')):
                mask_prob_img = nib.load(pjoin(sub_ses_deriv, f'sub-{sub}_ses-{ses}_{mask_prob}.nii.gz'))
                prob = mask_prob_img.get_fdata()
                prob_thr = np.where(prob > 0.5, 1, 0)
                nib.save(nib.Nifti1Image(prob_thr.astype(np.int32), affine=mask_prob_img.affine), pjoin(sub_ses_deriv, f'sub-{sub}_ses-{ses}_{mask_t1}.nii.gz'))
            
    else:
        print('Register binary lesion mask to T1 space...')
        mask_t1 = mask_name.replace('FLAIR', 'T1w')
        for ses in [ses01, ses02, ses03]:
            subprocess.Popen(f"antsApplyTransforms -d 3 -i {pjoin(bids, 'derivatives', deriv_mask, f'sub-{sub}', f'ses-{ses}', f'sub-{sub}_ses-{ses}_{mask_name}.nii.gz')} -r {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses}', f'sub-{sub}_ses-{ses}_MPRAGE.nii.gz')} -n GenericLabel -t {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses}', f'sub-{sub}_ses-{ses}_FLAIR0GenericAffine.mat')} -o {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses}', f'sub-{sub}_ses-{ses}_{mask_t1}.nii.gz')}", shell=True).wait()
    
    # Register each binary lesion mask in the half-way space
    print('Register binary lesion mask to halfway space...')
    for ses in [ses02, ses03]:
        subprocess.Popen(f"mri_vol2vol --mov {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses}', f'sub-{sub}_ses-{ses}_{mask_t1}.nii.gz')} --targ {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses01}-{ses}', f'sub-{sub}_ses-{ses}_reg-half_MPRAGE.nii.gz')} --o {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses01}-{ses}', f'sub-{sub}_ses-{ses}_reg-half_{mask_t1}.nii.gz')} --lta {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses01}-{ses}', f'sub-{sub}_ses-{ses}_reg-half_MPRAGE.lta')} --interp nearest", shell=True).wait()
    
    # Compute SEL candidates
    print('Compute SEL candidates...')
    sub_ses01_date, sub_ses02_date, ses01_ses02_time = bids_SEL_candidates(bids, sub, ses01, ses02, deriv=deriv, mask=mask_t1, samseg=samseg)
    sub_ses01_date, sub_ses03_date, ses01_ses03_time = bids_SEL_candidates(bids, sub, ses01, ses03, deriv=deriv, mask=mask_t1, samseg=samseg)
    
    # Register SEL in each ses01 and ses03
    print('Register SEL back in each sessions...')
    for ses in [ses01, ses03]:
        if samseg:
            subprocess.Popen(f"mri_vol2vol --mov {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses}', f'sub-{sub}_ses-{ses}_{mask_t1}.nii.gz')} --targ {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses01}-{ses03}', f'SEL_candidates_samseg.nii.gz')} --o {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses}', f'sub-{sub}_ses-{ses}_SEL_candidates_samseg.nii.gz')} --lta {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses01}-{ses03}', f'sub-{sub}_ses-{ses}_reg-half_MPRAGE.lta')} --inv --interp nearest", shell=True).wait()
            subprocess.Popen(f"mri_vol2vol --mov {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses}', f'sub-{sub}_ses-{ses}_MPRAGE.nii.gz')} --targ {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses01}-{ses03}', f'SEL_jacobian_samseg.nii.gz')} --o {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses}', f'sub-{sub}_ses-{ses}_SEL_jacobian_samseg.nii.gz')} --lta {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses01}-{ses03}', f'sub-{sub}_ses-{ses}_reg-half_MPRAGE.lta')} --inv", shell=True).wait()
        else:
            subprocess.Popen(f"mri_vol2vol --mov {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses}', f'sub-{sub}_ses-{ses}_{mask_t1}.nii.gz')} --targ {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses01}-{ses03}', f'SEL_candidates.nii.gz')} --o {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses}', f'sub-{sub}_ses-{ses}_SEL_candidates.nii.gz')} --lta {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses01}-{ses03}', f'sub-{sub}_ses-{ses}_reg-half_MPRAGE.lta')} --inv --interp nearest", shell=True).wait()
            subprocess.Popen(f"mri_vol2vol --mov {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses}', f'sub-{sub}_ses-{ses}_MPRAGE.nii.gz')} --targ {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses01}-{ses03}', f'SEL_jacobian.nii.gz')} --o {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses}', f'sub-{sub}_ses-{ses}_SEL_jacobian.nii.gz')} --lta {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses01}-{ses03}', f'sub-{sub}_ses-{ses}_reg-half_MPRAGE.lta')} --inv", shell=True).wait()
        subprocess.Popen(f"mri_vol2vol --mov {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses}', f'sub-{sub}_ses-{ses}_MPRAGE.nii.gz')} --targ {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses01}-{ses03}', f'jacobian_norm.nii.gz')} --o {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses}', f'sub-{sub}_ses-{ses}_jacobian_norm.nii.gz')} --lta {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses01}-{ses03}', f'sub-{sub}_ses-{ses}_reg-half_MPRAGE.lta')} --inv", shell=True).wait()
        
    # Register SEL in ses02 
    #subprocess.Popen(f"mri_vol2vol --mov {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses02}', f'sub-{sub}_ses-{ses02}_{mask_t1}.nii.gz')} --targ {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses01}-{ses02}', f'SEL_candidates.nii.gz')} --o {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses02}', f'sub-{sub}_ses-{ses02}_SEL_candidates.nii.gz')} --lta {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses01}-{ses02}', f'sub-{sub}_ses-{ses02}_reg-half_MPRAGE.lta')} --inv --interp nearest", shell=True).wait()
    #subprocess.Popen(f"mri_vol2vol --mov {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses02}', f'sub-{sub}_ses-{ses02}_MPRAGE.nii.gz')} --targ {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses01}-{ses02}', f'SEL_jacobian.nii.gz')} --o {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses02}', f'sub-{sub}_ses-{ses02}_SEL_jacobian.nii.gz')} --lta {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses01}-{ses02}', f'sub-{sub}_ses-{ses02}_reg-half_MPRAGE.lta')} --inv", shell=True).wait()
    subprocess.Popen(f"mri_vol2vol --mov {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses02}', f'sub-{sub}_ses-{ses02}_MPRAGE.nii.gz')} --targ {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses01}-{ses02}', f'jacobian_norm.nii.gz')} --o {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses02}', f'sub-{sub}_ses-{ses02}_jacobian_norm.nii.gz')} --lta {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses01}-{ses02}', f'sub-{sub}_ses-{ses02}_reg-half_MPRAGE.lta')} --inv", shell=True).wait()
    # Register ses-03 to ses-02
    subprocess.Popen(f"antsRegistrationSyN.sh -d 3 -f {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses02}', f'sub-{sub}_ses-{ses02}_MPRAGE.nii.gz')} -m {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses03}', f'sub-{sub}_ses-{ses03}_MPRAGE.nii.gz')} -t r -o {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses02}', f'sub-{sub}_ses-{ses03}_reg-ses{ses02}_MPRAGE')}", shell=True).wait()
    if pexists(pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses02}', f'sub-{sub}_ses-{ses03}_reg-ses{ses02}_MPRAGEWarped.nii.gz')):
        os.rename(pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses02}', f'sub-{sub}_ses-{ses03}_reg-ses{ses02}_MPRAGEWarped.nii.gz'), pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses02}', f'sub-{sub}_ses-{ses03}_reg-ses{ses02}_MPRAGE.nii.gz'))
    else:
        print('[ERROR] while registering ses03 MPRAGE to ses02 MPRAGE')
        return
    
    # # apply transofrmation to SEL_candidates mask
    if samseg:
        subprocess.Popen(f"antsApplyTransforms -d 3 -i {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses03}', f'sub-{sub}_ses-{ses03}_SEL_candidates_samseg.nii.gz')} -r {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses02}', f'sub-{sub}_ses-{ses02}_{mask_t1}.nii.gz')} -n GenericLabel -t {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses02}', f'sub-{sub}_ses-{ses03}_reg-ses{ses02}_MPRAGE0GenericAffine.mat')} -o {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses02}', f'sub-{sub}_ses-{ses02}_SEL_candidates_samseg.nii.gz')}", shell=True).wait()
    else:
        subprocess.Popen(f"antsApplyTransforms -d 3 -i {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses03}', f'sub-{sub}_ses-{ses03}_SEL_candidates.nii.gz')} -r {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses02}', f'sub-{sub}_ses-{ses02}_{mask_t1}.nii.gz')} -n GenericLabel -t {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses02}', f'sub-{sub}_ses-{ses03}_reg-ses{ses02}_MPRAGE0GenericAffine.mat')} -o {pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses02}', f'sub-{sub}_ses-{ses02}_SEL_candidates.nii.gz')}", shell=True).wait()
    
    # # create the SEL_jacobian file 
    ses02_jacobian_img = nib.load(pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses02}', f'sub-{sub}_ses-{ses02}_jacobian_norm.nii.gz'))
    ses02_jacobian = ses02_jacobian_img.get_fdata()
    if samseg:
        sel_cand = nib.load(pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses02}', f'sub-{sub}_ses-{ses02}_SEL_candidates_samseg.nii.gz')).get_fdata()
    else:
        sel_cand = nib.load(pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses02}', f'sub-{sub}_ses-{ses02}_SEL_candidates.nii.gz')).get_fdata()
    ses02_jacobian[sel_cand == 0] = 0
    sel_jac_img = nib.Nifti1Image(ses02_jacobian, affine=ses02_jacobian_img.affine)
    if samseg:
        nib.save(sel_jac_img, pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses02}', f'sub-{sub}_ses-{ses02}_SEL_jacobian_samseg.nii.gz'))
    else:
        nib.save(sel_jac_img, pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses02}', f'sub-{sub}_ses-{ses02}_SEL_jacobian.nii.gz'))
    del ses02_jacobian
    
    # Correct SEL mask by ses binary lesion mask
    print('Correct SEL mask in each session...')
    for ses in [ses01, ses02, ses03]:
        if samseg:
            mask = nib.load(pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses}', f'sub-{sub}_ses-{ses}_{mask_t1}.nii.gz')).get_fdata()
            sel_mask_img = nib.load(pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses}', f'sub-{sub}_ses-{ses}_SEL_candidates_samseg.nii.gz'))
            sel_mask = sel_mask_img.get_fdata()
            sel_mask_img = nib.Nifti1Image(np.multiply(sel_mask, mask), affine=sel_mask_img.affine)
            nib.save(sel_mask_img, pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses}', f'sub-{sub}_ses-{ses}_SEL_candidates_samseg.nii.gz'))
            sel_jac_img = nib.load(pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses}', f'sub-{sub}_ses-{ses}_SEL_jacobian_samseg.nii.gz'))
            sel_jac = sel_jac_img.get_fdata()
            sel_jac_img = nib.Nifti1Image(np.multiply(sel_jac, mask), affine=sel_jac_img.affine)
            nib.save(sel_jac_img, pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses}', f'sub-{sub}_ses-{ses}_SEL_jacobian_samseg.nii.gz'))   
        else:
            mask = nib.load(pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses}', f'sub-{sub}_ses-{ses}_{mask_t1}.nii.gz')).get_fdata()
            sel_mask_img = nib.load(pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses}', f'sub-{sub}_ses-{ses}_SEL_candidates.nii.gz'))
            sel_mask = sel_mask_img.get_fdata()
            sel_mask_img = nib.Nifti1Image(np.multiply(sel_mask, mask), affine=sel_mask_img.affine)
            nib.save(sel_mask_img, pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses}', f'sub-{sub}_ses-{ses}_SEL_candidates.nii.gz'))
            sel_jac_img = nib.load(pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses}', f'sub-{sub}_ses-{ses}_SEL_jacobian.nii.gz'))
            sel_jac = sel_jac_img.get_fdata()
            sel_jac_img = nib.Nifti1Image(np.multiply(sel_jac, mask), affine=sel_jac_img.affine)
            nib.save(sel_jac_img, pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses}', f'sub-{sub}_ses-{ses}_SEL_jacobian.nii.gz'))        
        
    # Compute Concentricity
    print('Compute concentricity...')
    sel_concentricity = compute_concentricity(bids, sub, ses03, deriv=deriv, samseg=samseg)
    
    # Compute constancy
    print('Compute constancy...')
    sel_constancy = compute_constancy(bids, sub, [ses02, ses03], deriv=deriv, samseg=samseg)
    
    # merge lesion concentricity and constancy DF together and save them at sub level
    print('Merge and save results dataframe...')
    sub_df = pd.merge(sel_concentricity, sel_constancy, on=['label_id'], how='outer')
    
    sub_df.insert(0, 'subject', f'sub-{sub}')
    sub_df.insert(1, 'ses01', f'ses-{ses01}')
    sub_df.insert(2, 'ses01_date', sub_ses01_date)
    sub_df.insert(3, 'ses01_time', 0)
    sub_df.insert(4, 'ses02', f'ses-{ses02}')
    sub_df.insert(5, 'ses02_date', sub_ses02_date)
    sub_df.insert(6, 'ses02_time', ses01_ses02_time)
    sub_df.insert(7, 'ses03', f'ses-{ses03}')
    sub_df.insert(8, 'ses03_date', sub_ses03_date)
    sub_df.insert(9, 'ses03_time', ses01_ses03_time)
    
    samseg_tag = '_samseg' if samseg else ''
    
    sub_df.to_excel(pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'SEL_candidates{samseg_tag}.xlsx'), index=False)
    
    
    
def compute_concentricity(bids, sub, ses, deriv='SEL', samseg=False):
    sub_ses_deriv = pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses}')
    
    samseg_tag = '_samseg' if samseg else ''
    # check files
    if not pexists(pjoin(sub_ses_deriv, f'sub-{sub}_ses-{ses}_SEL_candidates{samseg_tag}.nii.gz')):
        print('[ERROR] no SEL candidates')
        return
    
    if not pexists(pjoin(sub_ses_deriv, f'sub-{sub}_ses-{ses}_SEL_jacobian{samseg_tag}.nii.gz')):
        print('[ERROR] no SEL jacobian')
        return
    
    mask = nib.load(pjoin(sub_ses_deriv, f'sub-{sub}_ses-{ses}_SEL_candidates{samseg_tag}.nii.gz')).get_fdata()
    jacobian = nib.load(pjoin(sub_ses_deriv, f'sub-{sub}_ses-{ses}_SEL_jacobian{samseg_tag}.nii.gz')).get_fdata()
    
    return concentricity(mask, jacobian)


def compute_constancy(bids, sub, sess, deriv='SEL', samseg=False):
    
    samseg_tag = '_samseg' if samseg else ''
    
    masks = []
    jacobians = []
    for ses in sess:
    
        sub_ses_deriv = pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses}')
        
        # check files
        if not pexists(pjoin(sub_ses_deriv, f'sub-{sub}_ses-{ses}_SEL_candidates{samseg_tag}.nii.gz')):
            print('[ERROR] no SEL candidates for ses', ses)
            return
        
        if not pexists(pjoin(sub_ses_deriv, f'sub-{sub}_ses-{ses}_SEL_jacobian{samseg_tag}.nii.gz')):
            print('[ERROR] no SEL jacobian for ses', ses)
            return
        
        mask = nib.load(pjoin(sub_ses_deriv, f'sub-{sub}_ses-{ses}_SEL_candidates{samseg_tag}.nii.gz')).get_fdata()
        jacobian = nib.load(pjoin(sub_ses_deriv, f'sub-{sub}_ses-{ses}_SEL_jacobian{samseg_tag}.nii.gz')).get_fdata()
        
        masks.append(mask)
        jacobians.append(jacobian)
    
    return constancy(masks, jacobians)


def compute_criteria_only(bids, sub, ses01, ses02, ses03, deriv='SEL', samseg=False):
    
    samseg_tag = '_samseg' if samseg else ''
    
    # Compute Concentricity
    print('Compute concentricity...')
    sel_concentricity = compute_concentricity(bids, sub, ses03, deriv=deriv, samseg=samseg)
    
    # Compute constancy
    print('Compute constancy...')
    sel_constancy = compute_constancy(bids, sub, [ses02, ses03], deriv=deriv, samseg=samseg)
    
    # merge lesion concentricity and constancy DF together and save them at sub level
    print('Merge and save results dataframe...')
    sub_df = pd.merge(sel_concentricity, sel_constancy, on=['label_id'], how='outer')
    
    sub_ses01_date, sub_ses02_date, ses01_ses02_time = get_ses_date_and_time(bids, sub, ses01, ses02)
    sub_ses01_date, sub_ses03_date, ses01_ses03_time = get_ses_date_and_time(bids, sub, ses01, ses03)
    
    sub_df.insert(0, 'subject', f'sub-{sub}')
    sub_df.insert(1, 'ses01', f'ses-{ses01}')
    sub_df.insert(2, 'ses01_date', sub_ses01_date)
    sub_df.insert(3, 'ses01_time', 0)
    sub_df.insert(4, 'ses02', f'ses-{ses02}')
    sub_df.insert(5, 'ses02_date', sub_ses02_date)
    sub_df.insert(6, 'ses02_time', ses01_ses02_time)
    sub_df.insert(7, 'ses03', f'ses-{ses03}')
    sub_df.insert(8, 'ses03_date', sub_ses03_date)
    sub_df.insert(9, 'ses03_time', ses01_ses03_time)
    
    sub_df.to_excel(pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'SEL_candidates{samseg_tag}.xlsx'), index=False)
    

def get_ses_date_and_time(bids, sub, ses01, ses02):
    
    # open pandas file for date
    print('check dates for ses01 and ses02...')
    participants_df = pd.read_csv(pjoin(bids, 'participants.tsv'), sep='\t')
    sub_ses01_df = participants_df[(participants_df['participant_id'] == f'sub-{sub}') & (participants_df['session'] == f'ses-{ses01}')]
    if sub_ses01_df.empty:
        print('[ERROR] sub_ses01 not found in participants.tsv')
        return
    
    sub_ses02_df = participants_df[(participants_df['participant_id'] == f'sub-{sub}') & (participants_df['session'] == f'ses-{ses02}')]
    if sub_ses02_df.empty:
        print('[ERROR] sub_ses02 not found in participants.tsv')
        return
    
    try:
        sub_ses01_date = pd.Timestamp(str(sub_ses01_df.at[sub_ses01_df.index[0], 'date']))
        sub_ses02_date = pd.Timestamp(str(sub_ses02_df.at[sub_ses02_df.index[0], 'date']))
        print((sub_ses02_date - sub_ses01_date))
        ses01_ses02_time = (sub_ses02_date - sub_ses01_date).days
    except Exception as e:
        print(f'[ERROR] {e} when computing ses02 to ses01 delta time')
        return
    
    return sub_ses01_date, sub_ses02_date, ses01_ses02_time


def get_session_list(bids, subj, ses_details):
    """Helper function to get the list of sessions for a given subject."""
    sess = []
    if ses_details == 'all':
        for d in os.listdir(pjoin(bids, f'sub-{subj}')):
            if d.startswith('ses-'):
                sess.append(d.split('-')[1])
    else:
        for s in ses_details.split(','):
            if '-' in s:
                s0, s1 = map(int, s.split('-'))
                for si in range(s0, s1 + 1):
                    si_str = str(si).zfill(2)
                    if os.path.isdir(pjoin(bids, f'sub-{subj}', f'ses-{si_str}')):
                        sess.append(si_str)
            else:
                if os.path.isdir(pjoin(bids, f'sub-{subj}', f'ses-{s}')):
                    sess.append(s)
    return sess

def process_subject_range(bids, sub_range, ses_details):
    """Helper function to process a range of subjects."""
    subjects_and_sessions = []
    sub0, sub1 = map(int, sub_range.split('-'))
    for subi in range(sub0, sub1 + 1):
        subi_str = str(subi).zfill(3)
        if not os.path.isdir(pjoin(bids, f'sub-{subi_str}')):
            continue
        sess = get_session_list(bids, subi_str, ses_details)
        subjects_and_sessions.append((subi_str, sess))
    return subjects_and_sessions

def find_subjects_and_sessions(bids, sub, ses):
    subjects_and_sessions = []

    if sub == 'all':
        # Process all subjects
        for dirs in os.listdir(bids):
            if dirs.startswith('sub-'):
                subj = dirs.split('-')[1]
                sess = get_session_list(bids, subj, ses)
                subjects_and_sessions.append((subj, sess))
    else:
        # Process specified subjects
        for sub_item in sub.split(','):
            if '-' in sub_item:
                subjects_and_sessions.extend(process_subject_range(bids, sub_item, ses))
            else:
                if not os.path.isdir(pjoin(bids, f'sub-{sub_item}')):
                    continue
                sess = get_session_list(bids, sub_item, ses)
                subjects_and_sessions.append((sub_item, sess))
    
    return sorted(subjects_and_sessions)


def find_subs(bids, sub):
    subs = []
    if sub == 'all':
        for dirs in os.listdir(bids):
            if dirs.startswith('sub-'):
                subj = dirs.split('-')[1]
                subs.append(subj)
    else:
        for sub_item in sub.split(','):
            if '-' in sub_item:
                sub0, sub1 = map(int, sub_item.split('-'))
                for subi in range(sub0, sub1 + 1):
                    subi_str = str(subi).zfill(3)
                    if not os.path.isdir(pjoin(bids, f'sub-{subi_str}')):
                        continue
                    subs.append(subi_str)
            else:
                if not os.path.isdir(pjoin(bids, f'sub-{sub_item}')):
                    continue
                subs.append(sub_item)
    return subs


if __name__ == '__main__':
    
    description = '''
bids_SEL:
    Compute SEL detection algorithm on 3 sessions
    '''
    
    usage = '\npython %(prog)s bids sub ses01 ses02 ses03 [OPTIONS]'
    
    parser = argparse.ArgumentParser(description=description, usage=usage)
    
    parser.add_argument('bids', type=str, help='path towards a bids formatted database')
    parser.add_argument('sub', type=str, help='sub ID or list of sub ID to process (e.g. 001,002). The keyword "all" will select all subjects of the database, while "-" allow to select subject ID in between two border (e.g. 001-010)')
    parser.add_argument('ses01', type=str, help='ses01 ID')
    parser.add_argument('ses02', type=str, help='ses02 ID')
    parser.add_argument('ses03', type=str, help='ses03 ID')
    parser.add_argument('--derivative', dest='deriv', type=str, help='derivative folder (default: SEL)', default='SEL', required=False)
    parser.add_argument('--flair', dest='flair', type=str, help='FLAIR image name in anat folder (default: FLAIR)', default='FLAIR', required=False)
    parser.add_argument('--mprage', dest='mprage', type=str, help='MPRAGE image name in anat folder (default: acq-MPRAGE_T1w)', default='acq-MPRAGE_T1w', required=False)
    parser.add_argument('--derivative-mask', dest='deriv_mask', type=str, help='derivative folder of a binary lesion mask (default: nnUNet)', default='nnUNet', required=False)
    parser.add_argument('--mask', dest='mask', type=str, help='bianry lesions mask name to check if SEL (default: mask-bin_algo-nnUNet_FLAIR)', default='mask-bin_algo-nnUNet_FLAIR', required=False)
    parser.add_argument('--samseg', dest='samseg', help='Run SAMSEG instead of the using a precomputed lesions mask (default: False)', action='store_const', const=True, default=False, required=False)
    parser.add_argument('--criteria-only', dest='criteria', help='compute only concentricity and constancy criteria', action='store_const', const=True, default=False, required=False)
    
    # Parse the arguments
    try:
        args = parser.parse_args()
    except SystemExit as e:
        # This block catches the SystemExit exception raised by argparse when required args are missing
        if e.code != 0:  # Non-zero code indicates an error
            parser.print_help()
        sys.exit(e.code)
        
    bids = args.bids
    
    subs = find_subs(bids, args.sub)
    
    # sess = args.ses.split(',')
    # if len(sess) != 2:
    #     print(f'{args.ses} not coform')
    #     sys.exit(0)
    # ses01 = sess[0]
    # ses02 = sess[1]
    
    for sub in subs:
        print(sub, args.ses01, args.ses02, args.ses03)
        
        if args.criteria:
            compute_criteria_only(bids, sub, args.ses01, args.ses02, args.ses03, deriv=args.deriv, samseg=args.samseg)
        
        else:
            bids_SEL(bids, sub, args.ses01, args.ses02, args.ses03, deriv=args.deriv, flair=args.flair, mprage=args.mprage, deriv_mask=args.deriv_mask, mask_name=args.mask, samseg=args.samseg)
    

