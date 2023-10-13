"""
File to select the moving image and return it with the corresponding label
"""

import os
import argparse

import json
import numpy as np
import nibabel as nib
import pandas as pd
import csv
import datetime


def select_random_sub(data_path, fx_im_path, mult_sessions, mov_contrast, out_file='../../reg_pairs.csv', append=1):
    """
    Select a random subject of the dataset that is different than the one of the fixed image.
    And write in a csv file the pair of subjects used (for reproducibility)
    """

    fixed_sub = (fx_im_path.split('_')[0]).split('/')[-1]

    sub_lst = [sub for sub in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, sub))]
    sub_lst.remove(fixed_sub)
    sub_lst_final = [sub for sub in sub_lst if sub.startswith('sub-')]
    moving_sub = np.random.choice(sub_lst_final)
    mov_im_path = os.path.join(data_path, moving_sub, 'anat', f'{moving_sub}_{mov_contrast}.nii.gz')
    mov_label_path = os.path.join(data_path, 'derivatives/labels', moving_sub, 'anat', f'{moving_sub}_{mov_contrast}_labels-disc-manual.nii.gz')

    res_summary = dict()
    res_summary['fixed_subject'] = fixed_sub
    res_summary['moving_subject'] = moving_sub

    # write header (only if append=False)
    if not append or not os.path.isfile(out_file):
        with open(out_file, 'w') as csvfile:
            header = ['Timestamp', 'fixed_subject', 'moving_subject']
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()

    # populate data
    with open(out_file, 'a') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        line = list()
        line.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))  # Timestamp
        for val in res_summary.keys():
            line.append(str(res_summary[val]))
        spamwriter.writerow(line)

    return mov_im_path, mov_label_path


def run_main(model_inference_specs, fx_im_path, data_path, mult_sessions=False,
             mov_contrast='T1w', inter_sub_csv=None):
    """
    Select a moving image
    Preprocess the fixed and moving images
    """

    if inter_sub_csv is None:
        mov_im_path, mov_label_path = select_random_sub(data_path, fx_im_path, mult_sessions, mov_contrast)
    else:
        df = pd.read_csv(inter_sub_csv)
        fixed_sub = (fx_im_path.split('_')[0]).split('/')[-1]
        mov_subject = df.loc[df['fixed_subject'] == fixed_sub]['moving_subject']
        mov_im_path = os.path.join(data_path, mov_subject, 'anat', f'{mov_subject}_{mov_contrast}.nii.gz')
        mov_label_path = os.path.join(data_path, 'derivatives/labels', mov_subject, 'anat', f'{mov_subject}_{mov_contrast}_labels-disc-manual.nii.gz')

    moving_nii = nib.load(mov_im_path)
    mov_label_nii = nib.load(mov_label_path)
    nib.save(moving_nii, os.path.join(f'moving_before_proc.nii.gz'))
    nib.save(mov_label_nii, os.path.join(f'moving_labels.nii.gz'))

if __name__ == "__main__":

    # parse the commandline
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-path', required=True, type=str,
                        help='path to the config file with the inference models specificities')

    # parameters to be specified by the user
    parser.add_argument('--fx-img-path', required=True, help='path to the fixed image')
    parser.add_argument('--data-path', required=True, help='path to whole dataset')

    # WARNING: THE INTER SUBJECT HAS NOT BEEN IMPLEMENTED FOR MULTI SESSIONS YET
    parser.add_argument('--mult-sessions', required=False, type=str, default='False',
                        help='boolean to determine if the dataset has multiple sessions for the subjects'
                             '(default False) {\'0\',\'1\', \'False\',\'True\'}')

    parser.add_argument('--mov-img-contrast', required=False, default='T1w',
                        help='contrast of the fixed image: one of {T1w, T2w, T2star}')

    parser.add_argument('--inter-sub-csv', required=False, type=str, default=None,
                        help='list of pair of subjects to use for the inter subject registration with a column '
                             'fixed_subject and a column moving_subject')

    args = parser.parse_args()

    with open(args.config_path) as config_file:
        model_inference_specs = json.load(config_file)

    run_main(model_inference_specs, args.fx_img_path, args.data_path, eval(args.mult_sessions), args.mov_img_contrast, args.inter_sub_csv)
