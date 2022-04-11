"""
File taking three segmentation images (paths) as input and computing the dice metric to evaluate
how well the images are registered.
The dice scores are saved in a csv file.
"""

import argparse
import os
import sys

import numpy as np
import nibabel as nib
import csv
import datetime


if __name__ == "__main__":

    # -------------------------------------------------------------------------------------------------------- #
    # ----                                       PARSER ARGUMENTS                                         ---- #
    # -------------------------------------------------------------------------------------------------------- #

    # parse command line
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=f'Evaluate the registration of two volumes')

    # path parameters
    p.add_argument('--fx-seg-path', required=True, help='path to the spinal cord segmentation of the fixed image')
    p.add_argument('--moving-seg-path', required=True, help='path to the spinal cord segmentation of the moving image')

    p.add_argument('--moving-label-reg-seg-path', required=True,
                   help='path to the spinal cord segmentation of the moving image registered using the labels')
    p.add_argument('--moving-pmj-reg-seg-path', required=False, default=None,
                   help='path to the spinal cord segmentation of the moving image registered using the labels and pmj')
    p.add_argument('--moving-sc-reg-seg-path', required=True,
                   help='path to the spinal cord segmentation of the moving image registered using the labels, pmj and sc segmentation')

    p.add_argument('--warped-seg-path', required=True, help='path to the spinal cord segmentation of the moved image')

    p.add_argument('--sub-id', required=True, help='id of the subject')

    p.add_argument('--out-file', required=False, default='metrics_on_sc_seg.csv',
                   help='path to csv summarizing the results obtained on the SC segmentation with different metrics')
    p.add_argument('--append', type=int, required=False, default=1, choices=[0, 1],
                   help="Append results as a new line in the output csv file instead of overwriting it.")

    arg = p.parse_args()

    # -------------------------------------------------------------------------------------------------------- #
    # ----                                      LOADING THE VOLUMES                                       ---- #
    # -------------------------------------------------------------------------------------------------------- #

    if len(arg.fx_seg_path.split('.')) > 1:
        fx_im = nib.load(arg.fx_seg_path)
    else:
        fx_im = nib.load(f'{arg.fx_seg_path}.nii.gz')

    if len(arg.moving_seg_path.split('.')) > 1:
        moving_im = nib.load(arg.moving_seg_path)
    else:
        moving_im = nib.load(f'{arg.moving_seg_path}.nii.gz')

    if len(arg.moving_label_reg_seg_path.split('.')) > 1:
        moving_label_reg = nib.load(arg.moving_label_reg_seg_path)
    else:
        moving_label_reg = nib.load(f'{arg.moving_label_reg_seg_path}.nii.gz')

    if arg.moving_pmj_reg_seg_path is not None:
        pmj_used = True
    else:
        pmj_used = False
    if pmj_used:
        if len(arg.moving_pmj_reg_seg_path.split('.')) > 1:
            moving_pmj_reg = nib.load(arg.moving_pmj_reg_seg_path)
        else:
            moving_pmj_reg = nib.load(f'{arg.moving_pmj_reg_seg_path}.nii.gz')

    if len(arg.moving_sc_reg_seg_path.split('.')) > 1:
        moving_sc_reg = nib.load(arg.moving_sc_reg_seg_path)
    else:
        moving_sc_reg = nib.load(f'{arg.moving_sc_reg_seg_path}.nii.gz')

    if len(arg.warped_seg_path.split('.')) > 1:
        moved_im = nib.load(arg.warped_seg_path)
    else:
        moved_im = nib.load(f'{arg.warped_seg_path}.nii.gz')

    fx_im_val = fx_im.get_fdata()
    moving_im_val = moving_im.get_fdata()
    moving_label_im_val = moving_label_reg.get_fdata()
    if pmj_used:
        moving_pmj_im_val = moving_pmj_reg.get_fdata()
    moving_sc_im_val = moving_sc_reg.get_fdata()
    moved_im_val = moved_im.get_fdata()

    # -------------------------------------------------------------------------------------------------------- #
    # ----                        COMPUTE METRICS BASED ON SC SEGMENTATION OVERLAP                        ---- #
    # -------------------------------------------------------------------------------------------------------- #

    # TP --> SC seg in moving and in fixed
    # FP --> SC seg in moving but not in fixed (background)
    # TN --> Background in moving and in fixed
    # FN --> Background in moving but not in fixed (sc seg)

    TP_label_moving = np.sum(moving_label_im_val[fx_im_val == 1])
    FP_label_moving = np.sum(moving_label_im_val[fx_im_val == 0])
    TN_tmp_label_moving = moving_label_im_val[fx_im_val == 0]
    TN_label_moving = len(np.ravel(TN_tmp_label_moving)) - np.sum(TN_tmp_label_moving)
    FN_tmp_label_moving = moving_label_im_val[fx_im_val == 1]
    FN_label_moving = len(np.ravel(FN_tmp_label_moving)) - np.sum(FN_tmp_label_moving)

    if pmj_used:
        TP_pmj_moving = np.sum(moving_pmj_im_val[fx_im_val == 1])
        FP_pmj_moving = np.sum(moving_pmj_im_val[fx_im_val == 0])
        TN_tmp_pmj_moving = moving_pmj_im_val[fx_im_val == 0]
        TN_pmj_moving = len(np.ravel(TN_tmp_pmj_moving)) - np.sum(TN_tmp_pmj_moving)
        FN_tmp_pmj_moving = moving_pmj_im_val[fx_im_val == 1]
        FN_pmj_moving = len(np.ravel(FN_tmp_pmj_moving)) - np.sum(FN_tmp_pmj_moving)

    TP_sc_moving = np.sum(moving_sc_im_val[fx_im_val == 1])
    FP_sc_moving = np.sum(moving_sc_im_val[fx_im_val == 0])
    TN_tmp_sc_moving = moving_sc_im_val[fx_im_val == 0]
    TN_sc_moving = len(np.ravel(TN_tmp_sc_moving)) - np.sum(TN_tmp_sc_moving)
    FN_tmp_sc_moving = moving_sc_im_val[fx_im_val == 1]
    FN_sc_moving = len(np.ravel(FN_tmp_sc_moving)) - np.sum(FN_tmp_sc_moving)

    TP_moving = np.sum(moving_im_val[fx_im_val == 1])
    FP_moving = np.sum(moving_im_val[fx_im_val == 0])
    TN_tmp_moving = moving_im_val[fx_im_val == 0]
    TN_moving = len(np.ravel(TN_tmp_moving)) - np.sum(TN_tmp_moving)
    FN_tmp_moving = moving_im_val[fx_im_val == 1]
    FN_moving = len(np.ravel(FN_tmp_moving)) - np.sum(FN_tmp_moving)

    TP_moved = np.sum(moved_im_val[fx_im_val == 1])
    FP_moved = np.sum(moved_im_val[fx_im_val == 0])
    TN_tmp_moved = moved_im_val[fx_im_val == 0]
    TN_moved = len(np.ravel(TN_tmp_moved)) - np.sum(TN_tmp_moved)
    FN_tmp_moved = moved_im_val[fx_im_val == 1]
    FN_moved = len(np.ravel(FN_tmp_moved)) - np.sum(FN_tmp_moved)

    nb_vox_moving = len(np.ravel(moving_im_val))
    nb_sc_vox_moving = np.sum(moving_im_val)
    nb_vox_moved = len(np.ravel(moved_im_val))
    nb_sc_vox_moved = np.sum(moved_im_val)

    # Dice --> (2 * TP) / ((FP + TP) + (TP + FN))
    dice_fx_moving = (2 * TP_moving) / (TP_moving + TP_moving + FP_moving + FN_moving)
    dice_fx_moved = (2 * TP_moved) / (TP_moved + TP_moved + FP_moved + FN_moved)
    dice_fx_label_moving = (2 * TP_label_moving) / (TP_label_moving + TP_label_moving + FP_label_moving + FN_label_moving)
    if pmj_used:
        dice_fx_pmj_moving = (2 * TP_pmj_moving) / (TP_pmj_moving + TP_pmj_moving + FP_pmj_moving + FN_pmj_moving)
    dice_fx_sc_moving = (2 * TP_sc_moving) / (TP_sc_moving + TP_sc_moving + FP_sc_moving + FN_sc_moving)

    # Sensitivity --> TP / (TP + FN)
    sens_fx_moving = TP_moving / (TP_moving + FN_moving)
    sens_fx_moved = TP_moved / (TP_moved + FN_moved)

    # Specificity --> TN / (TN + FP)
    spec_fx_moving = TN_moving / (TN_moving + FP_moving)
    spec_fx_moved = TN_moved / (TN_moved + FP_moved)

    # Accuracy --> (TP + TN) / (TP + FP + FN + TN)
    acc_fx_moving = (TP_moving + TN_moving) / nb_vox_moving
    acc_fx_moved = (TP_moved + TN_moved) / nb_vox_moved

    # Precision --> TP / (TP + FP)
    prec_fx_moving = TP_moving / nb_sc_vox_moving
    prec_fx_moved = TP_moved / nb_sc_vox_moved

    # Jaccard (IoU) --> TP / (FP + TP + FN)
    jacc_fx_moving = TP_moving / (TP_moving + FP_moving + FN_moving)
    jacc_fx_moved = TP_moved / (TP_moved + FP_moved + FN_moved)

    res_summary = dict()
    res_summary['subject'] = arg.sub_id
    res_summary['dice_before_registration'] = dice_fx_moving
    res_summary['dice_registration_label'] = dice_fx_label_moving
    if pmj_used:
        res_summary['dice_registration_pmj'] = dice_fx_pmj_moving
    res_summary['dice_registration_sc'] = dice_fx_sc_moving
    res_summary['dice_after_registration'] = dice_fx_moved
    res_summary['jaccard_before_registration'] = jacc_fx_moving
    res_summary['jaccard_after_registration'] = jacc_fx_moved
    res_summary['sensitivity_before_registration'] = sens_fx_moving
    res_summary['sensitivity_after_registration'] = sens_fx_moved
    res_summary['precision_before_registration'] = prec_fx_moving
    res_summary['precision_after_registration'] = prec_fx_moved
    res_summary['specificity_before_registration'] = spec_fx_moving
    res_summary['specificity_after_registration'] = spec_fx_moved
    res_summary['accuracy_before_registration'] = acc_fx_moving
    res_summary['accuracy_after_registration'] = acc_fx_moved

    # write header (only if append=False)
    if not arg.append or not os.path.isfile(arg.out_file):
        with open(arg.out_file, 'w') as csvfile:
            if pmj_used:
                header = ['Timestamp', 'Subject', 'Dice_before_registration', 'Dice_after_label_registration',
                          'Dice_after_pmj_registration', 'Dice_after_sc_registration',
                          'Dice_after_dl_models_registration', 'Jaccard_before', 'Jaccard_after',
                          'Sensitivity_before', 'Sensitivity_after', 'Precision_before', 'Precision_after',
                          'Specificity_before', 'Specificity_after', 'Accuracy_before', 'Accuracy_after']
            else:
                header = ['Timestamp', 'Subject', 'Dice_before_registration', 'Dice_after_label_registration',
                          'Dice_after_sc_registration', 'Dice_after_dl_models_registration',
                          'Jaccard_before', 'Jaccard_after',
                          'Sensitivity_before', 'Sensitivity_after', 'Precision_before', 'Precision_after',
                          'Specificity_before', 'Specificity_after', 'Accuracy_before', 'Accuracy_after']
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()

    # populate data
    with open(arg.out_file, 'a') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        line = list()
        line.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))  # Timestamp
        for val in res_summary.keys():
            line.append(str(res_summary[val]))
        spamwriter.writerow(line)

    sys.exit(0)
