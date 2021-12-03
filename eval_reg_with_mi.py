"""
File taking three images (fixed, moving, moved) and computing the (normalized) Mutual Information (MI) between these
images to evaluate the registration performance.
"""

import argparse
import os

import numpy as np
import nibabel as nib
import csv
import datetime
from scipy.stats import entropy


def detect_zero_padding(im):
    """
    Return the x_min, y_min, ... of the non zero-padded area
    """
    xy_plan = np.sum(im, axis=2)
    yz_plan = np.sum(im, axis=0)

    x_plan = np.sum(xy_plan, axis=1)
    x_min = np.argwhere(x_plan > 0)[0][0]
    x_max = np.argwhere(x_plan > 0)[-1][0]

    y_plan = np.sum(yz_plan, axis=1)
    y_min = np.argwhere(y_plan > 0)[0][0]
    y_max = np.argwhere(y_plan > 0)[-1][0]

    z_plan = np.sum(yz_plan, axis=0)
    z_min = np.argwhere(z_plan > 0)[0][0]
    z_max = np.argwhere(z_plan > 0)[-1][0]

    return x_min, y_min, z_min, x_max, y_max, z_max


def normalized_mutual_information(image0, image1, bins=100):
    r"""
    Function from scikit-image github
    Compute the normalized mutual information (NMI).
    It ranges from 1 (perfectly uncorrelated image values) to 2 (perfectly correlated image values,
    whether positively or negatively).
    Parameters
    ----------
    image0, image1 : ndarray
        Images to be compared. The two input images must have the same number
        of dimensions.
    bins : int or sequence of int, optional
        The number of bins along each axis of the joint histogram.
    Returns
    -------
    nmi : float
        The normalized mutual information between the two arrays, computed at
        the granularity given by ``bins``. Higher NMI implies more similar
        input images.
    """

    hist, bin_edges = np.histogramdd(
        [np.reshape(image0, -1), np.reshape(image1, -1)],
        bins=bins
    )

    # hist = hist[1:, 1:]

    H0 = entropy(np.sum(hist, axis=0))
    H1 = entropy(np.sum(hist, axis=1))
    H01 = entropy(np.reshape(hist, -1))

    return (H0 + H1) / H01


if __name__ == "__main__":

    # -------------------------------------------------------------------------------------------------------- #
    # ----                                       PARSER ARGUMENTS                                         ---- #
    # -------------------------------------------------------------------------------------------------------- #

    # parse command line
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=f'Evaluate the registration of two volumes')

    # path parameters
    p.add_argument('--fx-im-path', required=True, help='path to the fixed image')
    p.add_argument('--moving-im-path', required=True, help='path to the moving image')
    p.add_argument('--warped-im-path', required=True, help='path to the moved image')

    p.add_argument('--sub-id', required=True, help='id of the subject')

    p.add_argument('--out-file', required=False, default='nmi.csv',
                   help='path to csv summarizing the mutual information results')
    p.add_argument('--append', type=int, required=False, default=1, choices=[0, 1],
                   help="Append results as a new line in the output csv file instead of overwriting it.")

    arg = p.parse_args()

    # -------------------------------------------------------------------------------------------------------- #
    # ----                                      LOADING THE VOLUMES                                       ---- #
    # -------------------------------------------------------------------------------------------------------- #

    if len(arg.fx_im_path.split('.')) > 1:
        fx_im = nib.load(arg.fx_im_path)
    else:
        fx_im = nib.load(f'{arg.fx_im_path}.nii.gz')

    if len(arg.moving_im_path.split('.')) > 1:
        moving_im = nib.load(arg.moving_im_path)
    else:
        moving_im = nib.load(f'{arg.moving_im_path}.nii.gz')

    if len(arg.warped_im_path.split('.')) > 1:
        moved_im = nib.load(arg.warped_im_path)
    else:
        moved_im = nib.load(f'{arg.warped_im_path}.nii.gz')

    fx_im_val = fx_im.get_fdata()
    moving_im_val = moving_im.get_fdata()
    moved_im_val = moved_im.get_fdata()

    x_min, y_min, z_min, x_max, y_max, z_max = detect_zero_padding(moving_im_val)

    # Resize the images to remove the zero-padded part so it's not considered in the NMI computation
    fx_im_val = fx_im_val[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
    moving_im_val = moving_im_val[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
    moved_im_val = moved_im_val[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]

    # -------------------------------------------------------------------------------------------------------- #
    # ----                                       COMPUTE THE NMI                                          ---- #
    # -------------------------------------------------------------------------------------------------------- #

    nmi_fx_moving = normalized_mutual_information(fx_im_val, moving_im_val)
    nmi_fx_moved = normalized_mutual_information(fx_im_val, moved_im_val)
    nmi_moving_moved = normalized_mutual_information(moving_im_val, moved_im_val)

    perc_nmi_improvement = 100 * (nmi_fx_moved - nmi_fx_moving)/nmi_fx_moving

    res_summary = dict()
    res_summary['subject'] = arg.sub_id
    res_summary['nmi_before_registration'] = nmi_fx_moving
    res_summary['nmi_after_registration'] = nmi_fx_moved
    res_summary['nmi_between_moving_and_moved_images'] = nmi_moving_moved
    res_summary['perc_nmi_improvement_with_registration'] = np.round(perc_nmi_improvement, 2)

    # write header (only if append=False)
    if not arg.append or not os.path.isfile(arg.out_file):
        with open(arg.out_file, 'w') as csvfile:
            header = ['Timestamp', 'Subject', 'NMI_before_registration', 'NMI_after_registration', 'NMI_between_moving_and_moved_images', 'Percentage_nmi_improvement_registration']
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

