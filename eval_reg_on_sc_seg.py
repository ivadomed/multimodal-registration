"""
File taking three segmentation images (paths) as input and computing the dice metric to evaluate
how well the images are registered.
The dice scores are saved in a csv file.
"""

import argparse
import os

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
    p.add_argument('--warped-seg-path', required=True, help='path to the spinal cord segmentation of the moved image')

    p.add_argument('--sub-id', required=True, help='id of the subject')

    p.add_argument('--out-file', required=False, default='dice_score.csv',
                   help='path to csv summarizing the dice scores')
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

    if len(arg.warped_seg_path.split('.')) > 1:
        moved_im = nib.load(arg.warped_seg_path)
    else:
        moved_im = nib.load(f'{arg.warped_seg_path}.nii.gz')

    fx_im_val = fx_im.get_fdata()
    moving_im_val = moving_im.get_fdata()
    moved_im_val = moved_im.get_fdata()

    # -------------------------------------------------------------------------------------------------------- #
    # ----                   COMPUTE THE DICE SCORE REPRESENTING SC SEGMENTATION OVERLAP                  ---- #
    # -------------------------------------------------------------------------------------------------------- #

    dice_fx_moving = np.sum(moving_im_val[fx_im_val == 1]) * 2.0 / (np.sum(moving_im_val) + np.sum(fx_im_val))
    dice_fx_moved = np.sum(moved_im_val[fx_im_val == 1]) * 2.0 / (np.sum(moved_im_val) + np.sum(fx_im_val))
    dice_moving_moved = np.sum(moved_im_val[moving_im_val == 1]) * 2.0 / (np.sum(moved_im_val) + np.sum(moving_im_val))

    perc_dice_improvement = 100 * (dice_fx_moved - dice_fx_moving)/dice_fx_moving

    res_summary = dict()
    res_summary['subject'] = arg.sub_id
    res_summary['dice_before_registration'] = dice_fx_moving
    res_summary['dice_after_registration'] = dice_fx_moved
    res_summary['dice_between_moving_and_moved_images'] = dice_moving_moved
    res_summary['perc_dice_improvement_with_registration'] = np.round(perc_dice_improvement, 2)

    # write header (only if append=False)
    if not arg.append or not os.path.isfile(arg.out_file):
        with open(arg.out_file, 'w') as csvfile:
            header = ['Timestamp', 'Subject', 'Dice_before_registration', 'Dice_after_registration', 'Dice_between_moving_and_moved_images', 'Percentage_dice_improvement_registration']
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
