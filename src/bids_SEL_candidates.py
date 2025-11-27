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
import numpy as np
import nibabel as nib
import pandas as pd 
from skimage.filters import apply_hysteresis_threshold
from skimage.measure import label



def bids_SEL_candidates(bids, sub, ses01, ses02, deriv='SEL', mask='mask-bin_algo-nnUNet_T1w', samseg=False, JE1=0.125, JE2=0.04):
    
    sub_ses_half = pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses01}-{ses02}')
   
    # check files
    print('checking files...')
    if int(ses01) >= int(ses02):
        print('[ERROR] ses01 > ses02')
        return
    
    if JE1 < JE2:
        print('[ERROR] JE1 < JE2')
        return
    
    if not pexists(pjoin(sub_ses_half, f'sub-{sub}_ses-{ses01}_to_ses-{ses02}_jacobian.nii.gz')):
        print('[ERROR] missing ses01 to ses02 jacobian')
        return
    
    if not pexists(pjoin(sub_ses_half, f'sub-{sub}_ses-{ses02}_to_ses-{ses01}_jacobian.nii.gz')):
        print('[ERROR] missing ses02 to ses01 jacobian')
        return
    
    if not pexists(pjoin(sub_ses_half, f'sub-{sub}_ses-{ses02}_reg-half_{mask}.nii.gz')):
        print('[ERROR] no binary lesion mask')
        return
    
    if not pexists(pjoin(bids, 'participants.tsv')):
        print('[ERROR] missing BIDS participants.tsv')
        return
    
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
    
    # open lesion mask and jacobian
    print('Loading images...')
    mask_img = nib.load(pjoin(sub_ses_half, f'sub-{sub}_ses-{ses02}_reg-half_{mask}.nii.gz'))
    mask = mask_img.get_fdata()
    jacobian_img = nib.load(pjoin(sub_ses_half, f'sub-{sub}_ses-{ses01}_to_ses-{ses02}_jacobian.nii.gz'))
    jacobian = jacobian_img.get_fdata()
    
    if mask.shape != jacobian.shape:
        print('[ERROR] ses01 mask to jacobian shape mismatch')
        return
    
    if len(np.unique(mask)) > 2 :
        print('[WARNING] lesions not binary mask')
        print('binarizing mask ...')
        mask[mask != 0] = 1
    
    if ses01_ses02_time == 0:
        print('[ERROR] 0 days between ses01 to ses02')
    
    # normalize jacobian per year
    print('Normalize jacobian per year...')
    year_norm = ses01_ses02_time/365
    
    print(f'{year_norm=}')
    
    jacobian_norm = np.divide(jacobian, year_norm)
    
    SEL_cand, SEL_cand_jac = SEL_candidates(jacobian_norm, mask, JE1, JE2)
    
    SEL_cand_img = nib.Nifti1Image(SEL_cand, affine=mask_img.affine)
    if samseg:
        nib.save(SEL_cand_img, pjoin(sub_ses_half, 'SEL_candidates_samseg.nii.gz'))
    else:
        nib.save(SEL_cand_img, pjoin(sub_ses_half, 'SEL_candidates.nii.gz'))
    
    SEL_cand_jac_img = nib.Nifti1Image(SEL_cand_jac, affine=jacobian_img.affine)
    if samseg:
        nib.save(SEL_cand_jac_img, pjoin(sub_ses_half, 'SEL_jacobian_samseg.nii.gz'))
    else:
        nib.save(SEL_cand_jac_img, pjoin(sub_ses_half, 'SEL_jacobian.nii.gz'))
    
    jacobian_norm_img = nib.Nifti1Image(jacobian_norm, affine=jacobian_img.affine)
    nib.save(jacobian_norm_img, pjoin(sub_ses_half, 'jacobian_norm.nii.gz'))
    
    return sub_ses01_date, sub_ses02_date, ses01_ses02_time
    
    
def SEL_candidates(jacobian_norm, lesions, JE1, JE2, min_lesion_volume=14):
    
    SEL_cand_bin = apply_hysteresis_threshold(np.multiply(jacobian_norm, lesions), JE2, JE1)
    SEL_cand = label(SEL_cand_bin)
    
    for lab in SEL_cand:
        w = np.where(SEL_cand==lab)
        if len(w[0]) < min_lesion_volume:
            SEL_cand[w] = 0
    
    SEL_cand_jac = np.copy(jacobian_norm)
    SEL_cand_jac[np.where(SEL_cand==0)] = 0
    
    return SEL_cand, SEL_cand_jac


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


def find_subs(sub):
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
bids_SEL_candidates:
    Compute SEL candidates mask based jacobian deformation field and binary lesion mask
    '''
    
    usage = '\npython %(prog)s bids sub ses01 ses02 [OPTIONS]'
    
    parser = argparse.ArgumentParser(description=description, usage=usage)
    
    parser.add_argument('bids', type=str, help='path towards a bids formatted database')
    parser.add_argument('sub', type=str, help='sub ID or list of sub ID to process (e.g. 001,002). The keyword "all" will select all subjects of the database, while "-" allow to select subject ID in between two border (e.g. 001-010)')
    parser.add_argument('ses01', type=str, help='ses01 ID')
    parser.add_argument('ses02', type=str, help='ses02 ID')
    parser.add_argument('--derivative', '-d', dest='deriv', type=str, help='name of the derivative folder (default: SEL)', default='SEL', required=False)
    parser.add_argument('--mask', '-m', dest='mask', type=str, help='name of the mask (default: mask-bin_algo-nnUNet_T1w)', default='mask-bin_algo-nnUNet_T1w', required=False)
    parser.add_argument('--samseg', dest='samseg', help='use samseg mask and _samseg ext (default: False)', action='store_const', const=True, default=False, required=False)
    
    # Parse the arguments
    try:
        args = parser.parse_args()
    except SystemExit as e:
        # This block catches the SystemExit exception raised by argparse when required args are missing
        if e.code != 0:  # Non-zero code indicates an error
            parser.print_help()
        sys.exit(e.code)
        
    bids = args.bids
    
    subs = find_subs(args.sub)
    
    # sess = args.ses.split(',')
    # if len(sess) != 2:
    #     print(f'{args.ses} not coform')
    #     sys.exit(0)
    # ses01 = sess[0]
    # ses02 = sess[1]
    
    for sub in subs:
        print(sub, args.ses01, args.ses02)
        
        bids_SEL_candidates(bids, sub, args.ses01, args.ses02, deriv=args.deriv, mask=args.mask, samseg=args.samseg)
    

