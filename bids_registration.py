"""
File to load a trained registration model and register two images together.
It includes a preprocessing step to transform the volumes to the dimensions required by the model.
The images will be processed (and saved as im_name_proc) to be used in the registration model and then registered.
The registered image is saved (as im_name_proc_reg_to_CONTRAST). The deformation field is also saved.
This file can be used in a batch script to register the data of interest from all subjects in a dataset organized
according to the Brain Imaging Data Structure (BIDS) convention.
The contrast of the fixed image can be specified as argument for the files naming
"""

import os
import argparse

import numpy as np
import nibabel as nib
import voxelmorph as vxm

from nilearn.image import resample_img

def preprocess(im_nii, mov_im_nii, in_shape=(160, 160, 192)):
    ''' Resize and normalize image. '''

    rescale_factor = 1 / (np.max(im_nii.get_fdata()) - np.min(im_nii.get_fdata()))
    im_shape = im_nii.shape

    # Scale the data between 0 and 1
    im_nii.header.set_slope_inter(rescale_factor, - np.min(im_nii.get_fdata()))
    nib.save(im_nii, 'scaled_image.nii')
    scaled_img = nib.load('scaled_image.nii')

    # sampling strategy: https://www.kaggle.com/mechaman/resizing-reshaping-and-resampling-nifti-files
    target_shape = np.array(in_shape)
    new_resolution = [im_shape[0]/target_shape[0], im_shape[1]/target_shape[1], im_shape[2]/target_shape[2]]
    new_affine = np.zeros((4, 4))
    new_affine[:3, :3] = np.diag(new_resolution)
    # putting point 0,0,0 in the middle of the new volume - this could be refined in the future
    new_affine[:3, 3] = target_shape*new_resolution/2.*-1
    new_affine[3, 3] = 1.

    # Resample the 3d image to be in the dimension expected by the registration model
    resampled_nii = resample_img(scaled_img, target_affine=new_affine,
                                 target_shape=target_shape, interpolation='continuous')
    os.remove('scaled_image.nii')

    # Moving image
    rescale_factor = 1 / (np.max(mov_im_nii.get_fdata()) - np.min(mov_im_nii.get_fdata()))

    # Scale the data between 0 and 1
    mov_im_nii.header.set_slope_inter(rescale_factor, - np.min(mov_im_nii.get_fdata()))
    nib.save(mov_im_nii, 'scaled_image_mov.nii')
    scaled_img_mov = nib.load('scaled_image_mov.nii')

    # Resample the 3d image to be in the dimension expected by the registration model
    mov_resampled_nii = resample_img(scaled_img_mov, target_affine=new_affine,
                                     target_shape=target_shape, interpolation='continuous')
    os.remove('scaled_image_mov.nii')

    return resampled_nii, mov_resampled_nii


def register(reg_model, fx_im_path, mov_im_path, fx_contrast='T1w'):
    """
    Preprocess the two images using the provided model
    and register the moving image to the fixed one.
    Save the warped image and the deformation field.
    """

    model_in_shape = reg_model.inputs[0].shape[1:-1]

    fixed_nii = nib.load(f'{fx_im_path}.nii.gz')
    moving_nii = nib.load(f'{mov_im_path}.nii.gz')

    fixed, moving = preprocess(fixed_nii, moving_nii, model_in_shape)

    nib.save(fixed, os.path.join(f'{fx_im_path}_proc.nii.gz'))
    nib.save(moving, os.path.join(f'{mov_im_path}_proc.nii.gz'))

    moved, warp = reg_model.predict([np.expand_dims(moving.get_fdata().squeeze(), axis=(0, -1)),
                                     np.expand_dims(fixed.get_fdata().squeeze(), axis=(0, -1))])

    moved = nib.Nifti1Image(moved[0, ..., 0], fixed.affine)
    warp = nib.Nifti1Image(warp[0, ..., 0], fixed.affine)

    nib.save(moved, os.path.join(f'{mov_im_path}_proc_reg_to_{fx_contrast}.nii.gz'))
    nib.save(warp, os.path.join(f'{mov_im_path}_proc_field_to_{fx_contrast}.nii.gz'))


def run_main(reg_model_path, fx_im_path, mov_im_path, fx_im_contrast='T1w'):
    """
    Load the registration model
    Preprocess the fixed and moving images
    Register
    """
    # Load the registration model
    model = vxm.networks.VxmDense.load(reg_model_path, input_model=None)

    register(model, fx_im_path, mov_im_path, fx_contrast=fx_im_contrast)


if __name__ == "__main__":

    # parse the commandline
    parser = argparse.ArgumentParser()

    # parameters to be specified by the user
    parser.add_argument('--model-path', required=True, type=str, help='path to the registration model')

    parser.add_argument('--fx-img-path', required=True, help='path to the fixed image')
    parser.add_argument('--mov-img-path', required=True, help='path to the moving image')

    parser.add_argument('--fx-img-contrast', required=False, default='T1w',
                        help='contrast of the fixed image: one of {T1w, T2w, T2star}')

    args = parser.parse_args()

    run_main(args.model_path, args.fx_img_path, args.mov_img_path, args.fx_img_contrast)
