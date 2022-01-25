"""
File to load two trained registration models and register two images together using the models one after the other.
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
import tensorflow as tf
import tensorflow.keras.backend as K
import nibabel as nib
import voxelmorph as vxm

from nilearn.image import resample_img


def preprocess(im_nii, mov_im_nii, in_shape=(160, 160, 192)):
    ''' Resize and normalize image. '''

    # Scale the data between 0 and 1
    img = im_nii.get_fdata()
    scaled_img = (img - np.min(img)) / (np.max(img) - np.min(img))
    scaled_nii = nib.Nifti1Image(scaled_img, im_nii.affine)

    # sampling strategy: https://www.kaggle.com/mechaman/resizing-reshaping-and-resampling-nifti-files
    # the brain images are mapped to the specified target shape at 1 mm isotropic resolution
    target_shape = np.array(in_shape)
    new_resolution = [1, 1, 1]
    new_affine = np.zeros((4, 4))
    new_affine[:3, :3] = np.diag(new_resolution)
    # putting point 0,0,0 in the middle of the new volume - this could be refined in the future
    new_affine[:3, 3] = target_shape*new_resolution/2.*-1
    new_affine[3, 3] = 1.

    # Resample the 3d image to be in the dimension expected by the registration model
    resampled_nii = resample_img(scaled_nii, target_affine=new_affine,
                                 target_shape=target_shape, interpolation='continuous')

    # Scale the data between 0 and 1
    mov_img = mov_im_nii.get_fdata()
    scaled_mov_img = (mov_img - np.min(mov_img)) / (np.max(mov_img) - np.min(mov_img))
    scaled_nii = nib.Nifti1Image(scaled_mov_img, mov_im_nii.affine)

    # Resample the 3d image to be in the dimension expected by the registration model
    mov_resampled_nii = resample_img(scaled_nii, target_affine=new_affine,
                                     target_shape=target_shape, interpolation='continuous')

    return resampled_nii, mov_resampled_nii


def register(reg_model1, reg_model2, fx_im_path, mov_im_path, fx_contrast='T1w', already_preproc=0):
    """
    Preprocess the two images and register the moving image to the fixed one using the provided model.
    Save the warped image and the deformation field.
    """

    if already_preproc:
        fixed = nib.load(f'{fx_im_path}.nii.gz')
        moving = nib.load(f'{mov_im_path}.nii.gz')
    else:
        model_in_shape = reg_model1.inputs[0].shape[1:-1]

        fixed_nii = nib.load(f'{fx_im_path}.nii.gz')
        moving_nii = nib.load(f'{mov_im_path}.nii.gz')

        fixed, moving = preprocess(fixed_nii, moving_nii, model_in_shape)

        nib.save(fixed, os.path.join(f'{fx_im_path}_proc.nii.gz'))
        nib.save(moving, os.path.join(f'{mov_im_path}_proc.nii.gz'))

    moved_first_reg, warp_first_reg = reg_model1.predict([np.expand_dims(moving.get_fdata().squeeze(), axis=(0, -1)),
                                                          np.expand_dims(fixed.get_fdata().squeeze(), axis=(0, -1))])

    moved, warp_second_reg = reg_model2.predict([moved_first_reg,
                                                 np.expand_dims(fixed.get_fdata().squeeze(), axis=(0, -1))])

    warp = vxm.utils.compose([K.constant(warp_first_reg[0, ...]), K.constant(warp_second_reg[0, ...])])
    warp_arr = K.eval(warp)

    moved = nib.Nifti1Image(moved[0, ..., 0], fixed.affine)
    warp = nib.Nifti1Image(warp_arr, fixed.affine)

    nib.save(moved, os.path.join(f'{mov_im_path}_proc_reg_to_{fx_contrast}.nii.gz'))
    nib.save(warp, os.path.join(f'{mov_im_path}_proc_field_to_{fx_contrast}.nii.gz'))


def run_main(reg_model1_path, reg_model2_path, fx_im_path, mov_im_path, fx_im_contrast='T1w', already_preproc=0):
    """
    Load the registration model
    Preprocess the fixed and moving images
    Register
    """
    # Load the registration models
    model1 = vxm.networks.VxmDense.load(reg_model1_path, input_model=None)
    model2 = vxm.networks.VxmDense.load(reg_model2_path, input_model=None)

    register(model1, model2, fx_im_path, mov_im_path, fx_contrast=fx_im_contrast, already_preproc=already_preproc)


if __name__ == "__main__":

    # parse the commandline
    parser = argparse.ArgumentParser()

    # parameters to be specified by the user
    parser.add_argument('--model1-path', required=True, type=str,
                        help='path to the registration model (for affine registration)')
    parser.add_argument('--model2-path', required=True, type=str,
                        help='path to the registration model (for deformable registration)')

    parser.add_argument('--fx-img-path', required=True, help='path to the fixed image')
    parser.add_argument('--mov-img-path', required=True, help='path to the moving image')

    parser.add_argument('--fx-img-contrast', required=False, default='T1w',
                        help='contrast of the fixed image: one of {T1w, T2w, T2star}')

    parser.add_argument('--one-cpu-tf', required=False, type=str, default='True',
                        help='boolean to determine if the processes link to TF have access to one CPU (True) or all '
                             'the CPUs (False) {\'0\',\'1\', \'False\',\'True\'}')

    parser.add_argument('--already-preproc', type=int, required=False, default=0, choices=[0, 1],
                        help='Specify if the preprocessing step has already been done (1) or not (0)')

    args = parser.parse_args()

    if eval(args.one_cpu_tf):
        # set that TF can use only one CPU
        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess = tf.compat.v1.Session(config=session_conf)

    run_main(args.model1_path, args.model2_path, args.fx_img_path,
             args.mov_img_path, args.fx_img_contrast, args.already_preproc)