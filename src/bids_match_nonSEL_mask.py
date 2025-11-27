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
from scipy.spatial.distance import cdist


def bids_match_nonSEL_mask(bids, sub, ses01, ses02, ses03, deriv='SEL', mask='nonSEL'):
    
    subd = pjoin(bids, 'derivatives', deriv, f'sub-{sub}')
    
    pairs = [[ses01, ses02], [ses02, ses03]]
    
    for p in pairs:
        
        s1 = p[0]
        s2 = p[1]
        print('sessions:', s1, s2)
        if not pexists(pjoin(subd, f'ses-{s1}', f'sub-{sub}_ses-{s1}_{mask}.nii.gz')):
            print(f'[WARNING] no mask in ses-{s1}')
            continue
        if not pexists(pjoin(subd, f'ses-{s2}', f'sub-{sub}_ses-{s2}_{mask}.nii.gz')):
            print(f'[WARNING] no mask in ses-{s2}')
            continue
        
        if not pexists(pjoin(subd, f'ses-{s1}', f'sub-{sub}_ses-{s2}_reg-ses{s1}_MPRAGE0GenericAffine.mat')):
            print(f'perform registration between ses-{s2} and ses-{s1}')
            subprocess.Popen(f"antsRegistrationSyN.sh -d 3 -f {pjoin(subd, f'ses-{s1}', f'sub-{sub}_ses-{s1}_MPRAGE.nii.gz')} -m {pjoin(subd, f'ses-{s2}', f'sub-{sub}_ses-{s2}_MPRAGE.nii.gz')} -t r -o {pjoin(subd, f'ses-{s1}', f'sub-{sub}_ses-{s2}_reg-ses{s1}_MPRAGE')}", shell=True).wait()
            if pexists(pjoin(subd, f'ses-{s1}', f'sub-{sub}_ses-{s2}_reg-ses{s1}_MPRAGEWarped.nii.gz')):
                os.rename(pjoin(subd, f'ses-{s1}', f'sub-{sub}_ses-{s2}_reg-ses{s1}_MPRAGEWarped.nii.gz'), pjoin(subd, f'ses-{s1}', f'sub-{sub}_ses-{s2}_reg-ses{s1}_MPRAGE.nii.gz'))
            else:
                print('[ERROR during registration')
                return
            
        # apply registration on nonSEL mask from s1 to s2
        subprocess.Popen(f"antsApplyTransforms -d 3 -i {pjoin(subd, f'ses-{s1}', f'sub-{sub}_ses-{s1}_{mask}.nii.gz')} -r {pjoin(subd, f'ses-{s2}', f'sub-{sub}_ses-{s2}_MPRAGE.nii.gz')} -n GenericLabel -t [{pjoin(subd, f'ses-{s1}', f'sub-{sub}_ses-{s2}_reg-ses{s1}_MPRAGE0GenericAffine.mat')},1] -o {pjoin(subd, f'ses-{s2}', f'sub-{sub}_ses-{s1}_reg-{s2}_{mask}.nii.gz')}", shell=True).wait()
        
        # match lesions from s1 to s2
        mask_s2_img = nib.load(pjoin(subd, f'ses-{s2}', f'sub-{sub}_ses-{s2}_{mask}.nii.gz'))
        mask_s2 = mask_s2_img.get_fdata()
        mask_s1 = nib.load(pjoin(subd, f'ses-{s2}', f'sub-{sub}_ses-{s1}_reg-{s2}_{mask}.nii.gz')).get_fdata()
        
        mask_shape = mask_s2.shape
        lesions = list(np.unique(mask_s2))
        lesions.remove(0)
        if len(lesions) > 0:
            new_les = sorted(lesions, reverse=True)[0] + 1
        else:
            new_les = 1
        mask_s2_corr = np.zeros_like(mask_s2)
        for les in lesions:
            print(les)
            w = np.where(mask_s2 == les)
            les_size = len(w[0])
            ref_les = mask_s1[w]
            ref_u = list(np.unique(ref_les))
            ref_les_match = [(u,(len(np.where(ref_les==u)[0])/les_size)) for u in ref_u]
            ref_les_match = sorted(ref_les_match, key=lambda x: x[1], reverse=True)
            ref_les_match_thr = [e for e in ref_les_match if e[1] > 0.25]
            print(ref_les_match)
            
            if len(ref_les_match_thr) == 0:
                print('bruh WTF ?')
                
            elif len(ref_les_match_thr) == 1:
                if ref_les_match_thr[0][0] == 0:
                    print('new lesion')
                    mask_s2_corr[w] = new_les
                    new_les = new_les+1
                        
                else:
                    print('match lesion')
                    mask_s2_corr[w] = ref_les_match_thr[0][0]
                    
            else:
                ref_les_match_thr = [e for e in ref_les_match_thr if e[0] != 0]
                if len(ref_les_match_thr) == 1:
                    print('match lesion')
                    mask_s2_corr[w] = ref_les_match_thr[0][0]
                else:
                    print('separate lesion')
                    centroids = [(ref_les_match[u][0], np.array(np.mean(np.argwhere(mask_s1 == ref_les_match[u][0]), axis=0))) for u in range(len(ref_les_match)) if ref_les_match[u][0] != 0]
                    for v in range(len(w[0])):
                        voxel = (w[0][v], w[1][v], w[2][v])
                        voxel_n = [(voxel[0]+dx, voxel[1]+dy, voxel[2]+dz) for dx in [-2, -1, 0, 1, 2] for dy in [-1, -2, 0, 1, 2] for dz in [-2, -1, 0, 1, 2]]
                        voxel_nx = [nx for nx,_,_ in voxel_n if 0 <= nx < mask_shape[0]]
                        voxel_ny = [ny for _,ny,_ in voxel_n if 0 <= ny < mask_shape[1]]
                        voxel_nz = [nz for _,_,nz in voxel_n if 0 <= nz < mask_shape[2]]
                        voxel_w = tuple([np.array(voxel_nx), np.array(voxel_ny), np.array(voxel_nz)])
                        ref_v = mask_s1[voxel_w]
                        voxel_u = [(u, len(ref_v[np.where(ref_v == u)])) for u in ref_v]
                        voxel_u = sorted(voxel_u, key=lambda x: x[1], reverse=True)
                        if voxel_u[0][0] == 0 and len(voxel_u) == 1:
                            print(f'no matching for voxel {v}')
                            print('match to closest centroid')
                            voxel_centroid_dist = [(cent[0], cdist(voxel, cent[1])) for cent in centroids]
                            voxel_centroid_dist = sorted(voxel_centroid_dist, key=lambda x: x[1], reverse=True)
                            mask_s2_corr[voxel] = voxel_centroid_dist[0][0]
                        else:
                            print(f'match voxel {v} to closest neighbor')
                            mask_s2_corr[voxel] = voxel_u[0][0]
                            
        nib.save(nib.Nifti1Image(mask_s2_corr, affine=mask_s2_img.affine), pjoin(subd, f'ses-{s2}', f'sub-{sub}_ses-{s2}_{mask}.nii.gz'))
            

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
bids_match_nonSEL_mask:
    match nonSEL mask between session
    '''
    
    usage = '\npython %(prog)s bids sub ses01 ses02 ses03 [OPTIONS]'
    
    parser = argparse.ArgumentParser(description=description, usage=usage)
    
    parser.add_argument('bids', type=str, help='path towards a bids formatted database')
    parser.add_argument('sub', type=str, help='sub ID or list of sub ID to process (e.g. 001,002). The keyword "all" will select all subjects of the database, while "-" allow to select subject ID in between two border (e.g. 001-010)')
    parser.add_argument('ses01', type=str, help='ses01 ID')
    parser.add_argument('ses02', type=str, help='ses02 ID')
    parser.add_argument('ses03', type=str, help='ses03 ID')
    parser.add_argument('--derivative', '-d', dest='deriv', type=str, help='name of the derivative folder (default: SEL)', default='SEL', required=False)
    parser.add_argument('--mask', dest='mask', type=str, help='mask name to correct (default: nonSEL)', default='nonSEL', required=False)
    
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
        print(sub, args.ses01, args.ses02, args.ses03)
        
        bids_match_nonSEL_mask(bids, sub, args.ses01, args.ses02, args.ses03, deriv=args.deriv, mask=args.mask)
    

