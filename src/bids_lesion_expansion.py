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



def bids_lesion_expansion(bids, sub, ses01, ses02, mprage='acq-MPRAGE_T1w', flair='FLAIR', deriv='lesion_expansion'):
    
    # check files
    print('check files...')
    sub_ses01_anat = pjoin(bids, f'sub-{sub}', f'ses-{ses01}', 'anat')
    sub_ses02_anat = pjoin(bids, f'sub-{sub}', f'ses-{ses02}', 'anat')
    
    if int(ses01) >= int(ses02):
        print('[ERROR] ses01 >= ses02')
        return
    
    if not pexists(pjoin(sub_ses01_anat, f'sub-{sub}_ses-{ses01}_{mprage}.nii.gz')):
        print('[ERROR] sub-{sub} ses-{ses01} has no MPRAGE')
        return
    
    if not pexists(pjoin(sub_ses01_anat, f'sub-{sub}_ses-{ses01}_{flair}.nii.gz')):
        print('[ERROR] sub-{sub} ses-{ses01} has no FLAIR')
        return
    
    if not pexists(pjoin(sub_ses02_anat, f'sub-{sub}_ses-{ses02}_{mprage}.nii.gz')):
        print('[ERROR] sub-{sub} ses-{ses02} has no MPRAGE')
        return
    
    if not pexists(pjoin(sub_ses02_anat, f'sub-{sub}_ses-{ses02}_{flair}.nii.gz')):
        print('[ERROR] sub-{sub} ses-{ses02} has no FLAIR')
        return
    
    # preproc each session individually
    print('1. preprocessing')
    sub_ses01_deriv = pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses01}')
    sub_ses02_deriv = pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses02}')
    
    os.makedirs(sub_ses01_deriv, exist_ok=True)
    os.makedirs(sub_ses02_deriv, exist_ok=True)
    
    preprocessing(pjoin(sub_ses01_anat, f'sub-{sub}_ses-{ses01}_{mprage}.nii.gz'), pjoin(sub_ses01_anat, f'sub-{sub}_ses-{ses01}_{flair}.nii.gz'), preproc_out=sub_ses01_deriv, out_mprage=f'sub-{sub}_ses-{ses01}_MPRAGE', out_flair=f'sub-{sub}_ses-{ses01}_FLAIR', denoise=True, skullstrip=True, registration=True, reg_flair=True)
    preprocessing(pjoin(sub_ses02_anat, f'sub-{sub}_ses-{ses02}_{mprage}.nii.gz'), pjoin(sub_ses02_anat, f'sub-{sub}_ses-{ses02}_{flair}.nii.gz'), preproc_out=sub_ses02_deriv, out_mprage=f'sub-{sub}_ses-{ses02}_MPRAGE', out_flair=f'sub-{sub}_ses-{ses02}_FLAIR', denoise=True, skullstrip=True, registration=True, reg_flair=True)
    
    # register in an halfway space
    print('2. register MPRAGE images to halfway space')
    sub_ses_half = pjoin(bids, 'derivatives', deriv, f'sub-{sub}', f'ses-{ses01}-{ses02}')
    os.makedirs(sub_ses_half, exist_ok=True)
    mri1_to_mri2_lta = pjoin(sub_ses_half, f'MPRAGE{ses01}_to_MPRAGE{ses02}.lta')
    mri1_halfway = pjoin(sub_ses_half, f'sub-{sub}_ses-{ses01}_reg-half_MPRAGE')
    mri2_halfway = pjoin(sub_ses_half, f'sub-{sub}_ses-{ses02}_reg-half_MPRAGE')
    
    subprocess.Popen(f"mri_robust_register --mov {pjoin(sub_ses01_deriv, f'sub-{sub}_ses-{ses01}_MPRAGE.nii.gz')} --dst {pjoin(sub_ses02_deriv, f'sub-{sub}_ses-{ses02}_MPRAGE.nii.gz')} --lta {mri1_to_mri2_lta} --halfmov {mri1_halfway}.nii.gz --halfdst {mri2_halfway}.nii.gz --halfmovlta {mri1_halfway}.lta --halfdstlta {mri2_halfway}.lta --iscale --satit", shell=True).wait()
    
    # mv FLAIR images in halfway space
    print('3. mv FLAIR in halfway space')
    mri1_t2w_halfway = pjoin(sub_ses_half, f'sub-{sub}_ses-{ses01}_reg-half_FLAIR')
    mri2_t2w_halfway = pjoin(sub_ses_half, f'sub-{sub}_ses-{ses02}_reg-half_FLAIR')
    subprocess.Popen(f"mri_vol2vol --mov {pjoin(sub_ses01_deriv, f'sub-{sub}_ses-{ses01}_FLAIR.nii.gz')} --targ {mri1_halfway}.nii.gz --o {mri1_t2w_halfway}.nii.gz --lta {mri1_halfway}.lta", shell=True).wait()
    subprocess.Popen(f"mri_vol2vol --mov {pjoin(sub_ses02_deriv, f'sub-{sub}_ses-{ses02}_FLAIR.nii.gz')} --targ {mri2_halfway}.nii.gz --o {mri2_t2w_halfway}.nii.gz --lta {mri2_halfway}.lta", shell=True).wait()
    
    # mv mask
    print('4. mv brain mask in halfway space')
    subprocess.Popen(f"mri_vol2vol --mov {pjoin(sub_ses01_deriv, f'sub-{sub}_ses-{ses01}_MPRAGE_bet-mask.nii.gz')} --targ {mri1_halfway}.nii.gz --o {mri1_halfway}_bet-mask.nii.gz --lta {mri1_halfway}.lta --nearest", shell=True).wait()
    subprocess.Popen(f"mri_vol2vol --mov {pjoin(sub_ses02_deriv, f'sub-{sub}_ses-{ses02}_MPRAGE_bet-mask.nii.gz')} --targ {mri2_halfway}.nii.gz --o {mri2_halfway}_bet-mask.nii.gz --lta {mri2_halfway}.lta --nearest", shell=True).wait()
    
    mri1_mask_img = nib.load(f'{mri1_halfway}_bet-mask.nii.gz')
    mri1_mask_data = mri1_mask_img.get_fdata()
    mri2_mask_data = nib.load(f'{mri2_halfway}_bet-mask.nii.gz').get_fdata()
    
    # mask_union = union_masks(mri1_mask_data, mri2_mask_data)
    mask_union_img = nib.Nifti1Image(union_masks(mri1_mask_data, mri2_mask_data), affine=mri1_mask_img.affine, header=mri1_mask_img.header)
    brain_mask_half = pjoin(sub_ses_half, 'halfay_bet-mask.nii.gz')
    nib.save(mask_union_img, brain_mask_half)
    
    # non-linear registration
    print('5. non-linear registration')
    mri1_to_mri2_deformation_field = pjoin(sub_ses_half, f'sub_sub_ses-{ses01}_to_ses-{ses02}_field')
    subprocess.Popen(f"ANTS 3 -i 50x50x30 -r Gauss[2.0,0.0] -t SyN[0.7] -x {brain_mask_half} -m CC[{mri1_halfway}.nii.gz,{mri2_halfway}.nii.gz,1,4] -m CC[{mri1_t2w_halfway}.nii.gz,{mri2_t2w_halfway}.nii.gz,1,4] -o {mri1_to_mri2_deformation_field}", shell=True).wait()
    
    # Compute Jaobian
    print('6. compute jacobian')
    mri1_to_mri2_jacobian = pjoin(sub_ses_half, f'sub-{sub}_ses-{ses01}_to_ses-{ses02}_jacobian')
    mri2_to_mri1_jacobian = pjoin(sub_ses_half, f'sub-{sub}_ses-{ses02}_to_ses-{ses01}_jacobian')
    subprocess.Popen(f'CreateJacobianDeterminantImage 3 {mri1_to_mri2_deformation_field}Warp.nii.gz {mri1_to_mri2_jacobian}_centered-1.nii.gz', shell=True).wait()
    
    jacobian_centered1_img = nib.load(f'{mri1_to_mri2_jacobian}_centered-1.nii.gz')
    jacobian_centered1 = jacobian_centered1_img.get_fdata()
    jacobian = jacobian_centered1 - 1
    jacobian_img = nib.Nifti1Image(jacobian, affine=jacobian_centered1_img.affine, header=jacobian_centered1_img.header)
    nib.save(jacobian_img, f'{mri1_to_mri2_jacobian}.nii.gz')
    inv_jacobian_img = nib.Nifti1Image(-jacobian, affine=jacobian_centered1_img.affine, header=jacobian_centered1_img.header)
    nib.save(inv_jacobian_img, f'{mri2_to_mri1_jacobian}.nii.gz')
    
    # register deformation map back into the different session
    print('7. register deformation maps back to the different sessions')
    subprocess.Popen(f"mri_vol2vol --mov {pjoin(sub_ses01_deriv, f'sub-{sub}_ses-{ses01}_MPRAGE.nii.gz')} --targ {mri1_to_mri2_jacobian}.nii.gz --o {pjoin(sub_ses01_deriv, f'sub-{sub}_ses-{ses01}_to_ses-{ses02}_jacobian.nii.gz')} --lta {mri1_halfway}.lta --inv", shell=True).wait()
    subprocess.Popen(f"mri_vol2vol --mov {pjoin(sub_ses02_deriv, f'sub-{sub}_ses-{ses02}_MPRAGE.nii.gz')} --targ {mri2_to_mri1_jacobian}.nii.gz --o {pjoin(sub_ses02_deriv, f'sub-{sub}_ses-{ses02}_to_ses-{ses01}_jacobian.nii.gz')} --lta {mri2_halfway}.lta --inv", shell=True).wait()


def preprocessing(mprage_f, flair_f, preproc_out=None, out_mprage=None, out_flair=None, denoise=True, skullstrip=True, registration=True, reg_flair=False):
    
    if not pexists(mprage_f):
        print(f'[ERROR] {mprage_f} not found !')
        return 
    
    if not pexists(flair_f):
        print(f'[ERROR] {flair_f} not found !')
        return
    
    if preproc_out == None:
        flair_folder = flair_f.replace(flair_f.split(os.sep)[-1], '')
        preproc_out = pjoin(flair_folder, 'preprocessing')
        
        
    if not pexists(preproc_out):
        os.makedirs(preproc_out)
        
    mprage_name = mprage_f.split(os.sep)[-1]
    flair_name = flair_f.split(os.sep)[-1]
    
    if not out_mprage:
        out_mprage = mprage_name.replace('.nii.gz', '')
    
    if not out_flair:
        out_flair = flair_name.replace('.nii.gz', '')
    
    if denoise:
        
        mprage_f = preproc_denoise(mprage_f, pjoin(preproc_out, f'{out_mprage}.nii.gz'))
        flair_f = preproc_denoise(flair_f, pjoin(preproc_out, f'{out_flair}.nii.gz'))
    
    if skullstrip:
        
        mprage_f, mprage_bet_mask = preproc_skullstrip(mprage_f, pjoin(preproc_out, f'{out_mprage}.nii.gz'), pjoin(preproc_out, f'{out_mprage}_bet-mask.nii.gz'))
        flair_f, flair_bet_mask = preproc_skullstrip(flair_f, pjoin(preproc_out, f'{out_flair}.nii.gz'), pjoin(preproc_out, f'{out_flair}_bet-mask.nii.gz'))
    
    if registration:
        
        if reg_flair:
            
            flair_f = preproc_registration(flair_f, mprage_f, pjoin(preproc_out, out_flair.replace('.nii.gz', '_reg-MPRAGE')))
        
        else:
            
            mprage_f = preproc_registration(mprage_f, flair_f, pjoin(preproc_out,out_mprage.replace('.nii.gz', '_reg-FLAIR')))
    
        
    return mprage_f, flair_f


def preprocessing_mprage(mprage_f, preproc_out=None, out_mprage=None, denoise=True, skullstrip=True, registration=True, reg_flair=False):
    
    if not pexists(mprage_f):
        print(f'[ERROR] {mprage_f} not found !')
        return 
    
    if preproc_out == None:
        mprage_folder = mprage_f.replace(mprage_f.split(os.sep)[-1], '')
        preproc_out = pjoin(mprage_folder, 'preprocessing')
        
        
    if not pexists(preproc_out):
        os.makedirs(preproc_out)
        
    mprage_name = mprage_f.split(os.sep)[-1]
    
    if not out_mprage:
        out_mprage = mprage_name.replace('.nii.gz', '')
    
    if denoise:
        
        mprage_f = preproc_denoise(mprage_f, pjoin(preproc_out, f'{out_mprage}.nii.gz'))
    
    if skullstrip:
        
        mprage_f, mprage_bet_mask = preproc_skullstrip(mprage_f, pjoin(preproc_out, f'{out_mprage}.nii.gz'), pjoin(preproc_out, f'{out_mprage}_bet-mask.nii.gz'))
        
    return mprage_f

    
    
def preproc_denoise(img_f, out_f=None):
    
    if not pexists(img_f):
        print(f'[ERROR] {img_f} does not exists')
        return
    
    if out_f == None:
        out_f = img_f.replace('.nii.gz', '_den-ants.nii.gz')
        
    subprocess.Popen(f"DenoiseImage -d 3 -i {img_f} -o {out_f}", shell=True).wait()
    
    return out_f



def preproc_skullstrip(img_f, out_f=None, out_mask=None):
    
    if not pexists(img_f):
        print(f'[ERROR] {img_f} does not exists')
        return
    
    if out_f == None:
        out_f = img_f.replace('.nii.gz', '_bet-synthstrip.nii.gz')
        
    if out_mask == None:
        out_mask = img_f.replace('.nii.gz', '_bet-mask.nii.gz')
        
    subprocess.Popen(f"mri_synthstrip -i {img_f} -o {out_f} -m {out_mask}", shell=True).wait()
    
    return out_f, out_mask



def preproc_registration(moving_f, fixed_f, reg_f):
    
    if not pexists(moving_f):
        print(f'[ERROR] {moving_f} does not exists')
        return
    
    if not pexists(fixed_f):
        print(f'[ERROR] {fixed_f} does not exists')
        return
    
    subprocess.Popen(f"antsRegistrationSyN.sh -d 3 -f {fixed_f} -m {moving_f} -t r -o {reg_f}", shell=True).wait()
    
    if pexists(f'{reg_f}Warped.nii.gz'):
        os.rename(f'{reg_f}Warped.nii.gz', f'{reg_f}.nii.gz')
        
    return f'{reg_f}.nii.gz'


def union_masks(mask1, mask2):
    assert mask1.shape == mask2.shape, '[ERROR] The shapes of the two masks are not the same'
    union = np.zeros_like(mask1)
    union[np.where(mask1 != 0)] = 1
    union[np.where(mask2 != 0)] = 1
    return union



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
bids_lesion_expansion:
    Compute local lesion expansion based on non-linear deformation
    '''
    
    usage = '\npython %(prog)s bids sub ses01 ses02 [OPTIONS]'
    
    parser = argparse.ArgumentParser(description=description, usage=usage)
    
    parser.add_argument('bids', type=str, help='path towards a bids formatted database')
    parser.add_argument('sub', type=str, help='sub ID or list of sub ID to process (e.g. 001,002). The keyword "all" will select all subjects of the database, while "-" allow to select subject ID in between two border (e.g. 001-010)')
    parser.add_argument('ses01', type=str, help='ses01 ID')
    parser.add_argument('ses02', type=str, help='ses02 ID')
    parser.add_argument('--mprage', '-m', dest='mprage', type=str, help='name of the raw mprage sequence from anat folder (default: acq-MPRAGE_T1w)', default='acq-MPRAGE_T1w', required=False)
    parser.add_argument('--flair', '-f', dest='flair', type=str, help='name of the raw flair sequence from anat folder (default: FLAIR)', default='FLAIR', required=False)
    parser.add_argument('--derivative', '-d', dest='deriv', type=str, help='name of the derivative folder (default: lesion_separation)', default='lesion_expansion', required=False)
    
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
        
        bids_lesion_expansion(bids, sub, args.ses01, args.ses02, mprage=args.mprage, flair=args.flair, deriv=args.deriv)
    

