#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 09:32:58 2025

@author: colin
"""

import os
from os.path import join as pjoin
from os.path import exists as pexists
import sys
import argparse
import pandas as pd


def compile_SEL_resulsts(bids, subs, deriv='SEL', sel_file='SEL_candidates.xlsx'):
    
    sel_df = pd.DataFrame()
    for sub in subs:
        print(sub)
        sub_df = pd.read_excel(pjoin(bids, 'derivatives', deriv, f'sub-{sub}', sel_file))
        sel_df = pd.concat([sel_df, sub_df], ignore_index=True)
    
    # compute z-scores and define SELs
    sel_df['concentricity_zscore'] = (sel_df['concentricity'] - sel_df['concentricity'].mean())/sel_df['concentricity'].std()
    sel_df['constancy_zscore'] = (sel_df['constancy'] - sel_df['constancy'].mean())/sel_df['constancy'].std()
    sel_df['S'] = (sel_df['concentricity_zscore'] - sel_df['constancy_zscore'])
    sel_df['SEL'] = (sel_df['S'] > 0).astype(int)
    
    return sel_df


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
bids_SEL:
    Compute SEL detection algorithm on 3 sessions
    '''
    
    usage = '\npython %(prog)s bids sub ses01 ses02 ses03 [OPTIONS]'
    
    parser = argparse.ArgumentParser(description=description, usage=usage)
    
    parser.add_argument('bids', type=str, help='path towards a bids formatted database')
    parser.add_argument('sub', type=str, help='sub ID or list of sub ID to process (e.g. 001,002). The keyword "all" will select all subjects of the database, while "-" allow to select subject ID in between two border (e.g. 001-010)')
    parser.add_argument('--output', '-o', dest='output', type=str, help='path for the output excel file', required=True)
    parser.add_argument('--derivative', '-d', dest='deriv', type=str, help='derivative name (default: SEL)', default='SEL', required=False)
    parser.add_argument('--SEL-candidates', dest='sel_file', type=str, help='name of the SEL candidates excel file per subject (default: SEL_candidates.xlsx)', default='SEL_candidates.xlsx', required=False)
    
    
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
    subs = sorted(subs)
    # sess = args.ses.split(',')
    # if len(sess) != 2:
    #     print(f'{args.ses} not coform')
    #     sys.exit(0)
    # ses01 = sess[0]
    # ses02 = sess[1]
        
    sel_df = compile_SEL_resulsts(bids, subs, deriv=args.deriv, sel_file=args.sel_file)
    
    sel_df.to_excel(args.output, index=False)
