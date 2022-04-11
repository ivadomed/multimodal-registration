"""
File to align the PMJ of two 3D images, using a PMJ mask provided as input
"""

import argparse

import numpy as np
import nibabel as nib

def pmj_reg(fixed, moving_before_pmj_reg, fx_pmj_mask, mov_pmj_mask):
    """
    Align the pmj of the fixed and moving image.
    """

    from scipy.ndimage.interpolation import affine_transform

    pmj_mov_loc = np.where(mov_pmj_mask == np.max(mov_pmj_mask))
    pmj_mov_loc = [pmj_mov_loc[0][0], pmj_mov_loc[1][0], pmj_mov_loc[2][0]]

    pmj_fx_loc = np.where(fx_pmj_mask != 0)
    pmj_fx_loc = [pmj_fx_loc[0][0], pmj_fx_loc[1][0], pmj_fx_loc[2][0]]

    translation_vect_mov_to_fx = [- pmj_fx_loc[i] + pmj_mov_loc[i] for i in range(2)]  # Do the translation only for x and y
    translation_vect_mov_to_fx.append(0)

    mov_pmj_aligned = affine_transform(moving_before_pmj_reg.get_fdata(), np.identity(3), offset=translation_vect_mov_to_fx)
    mov_pmj_reg = nib.Nifti1Image(mov_pmj_aligned, fixed.affine)
    
    nib.save(mov_pmj_reg, 'moving_pmj_reg.nii.gz')


def register(fx_im_path, mov_im_path, fx_pmj_path, mov_pmj_path):
    """
    Align the two images using the pmj.
    """

    fixed = nib.load(fx_im_path)
    moving_before_pmj_reg = nib.load(mov_im_path)
    fx_pmj_mask = nib.load(fx_pmj_path)
    mov_pmj_mask = nib.load(mov_pmj_path)

    pmj_reg(fixed, moving_before_pmj_reg, fx_pmj_mask.get_fdata(), mov_pmj_mask.get_fdata())


def run_main(fx_im_path, mov_im_path, fx_pmj_path, mov_pmj_path):
    """
    Register the PMJ
    """
    register(fx_im_path, mov_im_path, fx_pmj_path, mov_pmj_path)


if __name__ == "__main__":

    # parse the commandline
    parser = argparse.ArgumentParser()

    # parameters to be specified by the user
    parser.add_argument('--fx-img-path', required=True, help='path to the fixed image')
    parser.add_argument('--mov-img-path', required=True, help='path to the moving image')

    parser.add_argument('--fx-pmj-path', required=True, help='path to the fixed pmj mask')
    parser.add_argument('--mov-pmj-path', required=True, help='path to the moving pmj mask')

    args = parser.parse_args()

    run_main(args.fx_img_path, args.mov_img_path, args.fx_pmj_path, args.mov_pmj_path)
