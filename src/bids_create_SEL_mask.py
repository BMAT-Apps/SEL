#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 14:28:26 2025

@author: colin
"""
import os 
from os.path import join as pjoin
from os.path import exists as pexists
import sys
import argparse
import subprocess
import pandas as pd
import nibabel as nib
import numpy as np
from skimage.measure import label


def bids_create_SEL_mask(bids, subs, SELxl, deriv='SEL', binary_mask='mask-bin_algo-nnUNet_T1w', samseg=False):
    
    # check excel file
    if not pexists(SELxl):
        print('[ERROR] SEL analysis excel file not found')
        return
    
    samseg_tag = '_samseg' if samseg else ''
    
    sel_df = pd.read_excel(SELxl)
    
    # subs = np.unique(list(sel_df['subject']))
    
    for sub in subs:
        # sub = sub.split('-')[1]
        print(sub)
        sub_df = sel_df[sel_df['subject'] == f'sub-{sub}']
        
        if sub_df.empty:
            if not pexists(pjoin(bids, 'derivatives', deriv, f'sub-{sub}')):
                continue
            sess = list(os.listdir(pjoin(bids, 'derivatives', deriv, f'sub-{sub}')))
            sess = [s.split('-')[1] for s in sess if 'ses-' in s and len(s) < 7]
            sess = sorted(sess)
            if len(sess) != 3:
                continue
            else:
                ses01 = sess[0]
                ses02 = sess[1]
                ses03 = sess[2]
        
        else:
            ses01 = sub_df.at[sub_df.index[0], 'ses01'].split('-')[1]
            ses02 = sub_df.at[sub_df.index[0], 'ses02'].split('-')[1]
            ses03 = sub_df.at[sub_df.index[0], 'ses03'].split('-')[1]
        
        for ses in [ses01, ses02, ses03]:
            
            print(ses)
            
            sub_ses_deriv = pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses}')
            
            # check files
            if not pexists(pjoin(sub_ses_deriv, f'sub-{sub}_ses-{ses}_SEL_candidates{samseg_tag}.nii.gz')):
                print(f'[ERROR] no ses-{ses} SEL candidates')
                return
            
            if not pexists(pjoin(sub_ses_deriv, f'sub-{sub}_ses-{ses}_{binary_mask}.nii.gz')):
                print(f'[ERROR] no ses-{ses} bin mask')
                return
            
            # SEL & SELpos
            sel_cand_img = nib.load(pjoin(sub_ses_deriv, f'sub-{sub}_ses-{ses}_SEL_candidates{samseg_tag}.nii.gz'))
            sel_cand = sel_cand_img.get_fdata()
            
            sel = np.zeros_like(sel_cand)
            selpos = np.zeros_like(sel_cand)
            for i in sub_df.index:
                lab = sub_df.at[i, 'label_id']
                if sub_df.at[i, 'SEL'] == 1:
                    sel[np.where(sel_cand == lab)] = lab
                elif sub_df.at[i, 'SEL'] == 0:
                    selpos[np.where(sel_cand == lab)] = lab
                else:
                    print('unrecognized SEL')
                    
            nib.save(nib.Nifti1Image(sel, affine=sel_cand_img.affine), pjoin(sub_ses_deriv, f'sub-{sub}_ses-{ses}_SEL{samseg_tag}.nii.gz'))
            nib.save(nib.Nifti1Image(selpos, affine=sel_cand_img.affine), pjoin(sub_ses_deriv, f'sub-{sub}_ses-{ses}_SELpos{samseg_tag}.nii.gz'))
            
            # nonSEL
            bin_mask_img = nib.load(pjoin(sub_ses_deriv, f'sub-{sub}_ses-{ses}_{binary_mask}.nii.gz'))
            bin_mask = bin_mask_img.get_fdata()
            blob_mask = label(bin_mask).astype(np.int32)
            nib.save(nib.Nifti1Image(blob_mask, affine=bin_mask_img.affine), pjoin(sub_ses_deriv, f'sub-{sub}_ses-{ses}_blob_mask.nii.gz'))
            blob_u = list(np.unique(blob_mask))
            blob_u.remove(0)
            for blob in blob_u:
                blob_w = np.where(blob_mask == blob)
                blob_sel_w = sel_cand[blob_w]
                blob_sel = (len(np.where(blob_sel_w != 0)[0]) / (len(blob_w[0])))
                if blob_sel > 0.3:
                    blob_mask[blob_w] = 0
                    
            nib.save(nib.Nifti1Image(blob_mask, affine=bin_mask_img.affine), pjoin(sub_ses_deriv, f'sub-{sub}_ses-{ses}_nonSEL{samseg_tag}.nii.gz'))

                
            


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
    parser.add_argument('--SEL-analysis', dest='SELxl', type=str, help='path towards the final SELs analysis excel file', required=True)
    parser.add_argument('--derivative', dest='deriv', type=str, help='derivative folder (default: SEL)', default='SEL', required=False)
    parser.add_argument('--binary-mask', dest='bin_mask', type=str, help='name of the binary mask to use (default: mask-bin_algo-nnUNet_T1w)', default='mask-bin_algo-nnUNet_T1w', required=False)
    parser.add_argument('--samseg', dest='samseg', help='To use the samseg mask (default: False)', action='store_const', const=True, default=False, required=False)

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
    
    # for sub in subs:
    #     print(sub, args.ses01, args.ses02, args.ses03)
        
    bids_create_SEL_mask(bids, subs, SELxl=args.SELxl, deriv=args.deriv, binary_mask=args.bin_mask, samseg=args.samseg)
        