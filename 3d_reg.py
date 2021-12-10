"""
File to load a trained registration model and register two images together.
It includes a preprocessing step to transform the volumes to the dimensions required by the model.
"""

import os
import argparse

import numpy as np
import nibabel as nib
import voxelmorph as vxm

from nilearn.image import resample_img


def preprocess(im_nii, in_shape=(160, 160, 192)):
    ''' Resize and normalize image. '''

    # Scale the data between 0 and 1
    img = im_nii.get_fdata()
    scaled_img = (img - np.min(img)) / (np.max(img) - np.min(img))
    scaled_nii = nib.Nifti1Image(scaled_img, im_nii.affine)

    # sampling strategy: https://www.kaggle.com/mechaman/resizing-reshaping-and-resampling-nifti-files
    target_shape = np.array(in_shape)
    new_resolution = [1, 1, 1]
    new_affine = np.zeros((4, 4))
    new_affine[:3, :3] = np.diag(new_resolution)
    # putting point 0,0,0 in the middle of the new volume - this could be refined in the future
    new_affine[:3, 3] = target_shape*new_resolution/2.*-1
    new_affine[3, 3] = 1.

    # Resample the 3d image to be in the dimension expected by the registration model
    resampled_nii = resample_img(scaled_nii, target_affine=new_affine,
                                 target_shape=target_shape, interpolation='nearest')

    return resampled_nii


def run_main(model_path, fx_im_path, mov_im_path, res_dir='res',
             out_im_path='warped_im', out_field_path='deform_field'):
    """
    Load a registration model, preprocess the two images and register the moving image to the fixed one.
    Save the warped image and the deformation field in the paths specified.
    """

    model = vxm.networks.VxmDense.load(model_path, input_model=None)

    reg_model = model
    model_in_shape = reg_model.inputs[0].shape[1:-1]

    fixed_nii = nib.load(f'{fx_im_path}.nii.gz')
    moving_nii = nib.load(f'{mov_im_path}.nii.gz')

    fixed = preprocess(fixed_nii, model_in_shape)
    moving = preprocess(moving_nii, model_in_shape)

    moved, warp = reg_model.predict([np.expand_dims(moving.get_fdata().squeeze(), axis=(0, -1)),
                                     np.expand_dims(fixed.get_fdata().squeeze(), axis=(0, -1))])

    nib.save(fixed, os.path.join(f'{fx_im_path}_preproc.nii.gz'))
    nib.save(moving, os.path.join(f'{mov_im_path}_preproc.nii.gz'))

    moved = nib.Nifti1Image(moved[0, ..., 0], fixed.affine)
    warp = nib.Nifti1Image(warp[0, ..., 0], fixed.affine)

    os.makedirs(res_dir, exist_ok=True)

    nib.save(moved, os.path.join(res_dir, f'{out_im_path}.nii.gz'))
    nib.save(warp, os.path.join(res_dir, f'{out_field_path}.nii.gz'))


if __name__ == "__main__":

    # parse the commandline
    parser = argparse.ArgumentParser()

    # parameters to be specified by the user
    parser.add_argument('--model-path', required=True, type=str, help='path to the registration model')

    parser.add_argument('--fx-img-path', required=True, help='path to the fixed image')
    parser.add_argument('--mov-img-path', required=True, help='path to the moving image')

    parser.add_argument('--res-dir', required=False, default='res', help='results output directory (default: res)')

    parser.add_argument('--out-img-name', required=False, default='warped_im',
                        help='name of the warped image that will result')
    parser.add_argument('--def-field-name', required=False, default='deform_field',
                        help='name of the deformation field that will result')

    args = parser.parse_args()

    run_main(args.model_path, args.fx_img_path, args.mov_img_path, args.res_dir, args.out_img_name, args.def_field_name)