"""
File to load a trained registration model and register two images together.
It includes a preprocessing step to scale the volumes and set it to an isotropic resolution of 1mm as required by the model.
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
import nibabel as nib
import voxelmorph as vxm

from nilearn.image import resample_img
from scipy.ndimage import zoom


def preprocess(data, im_nii, mov_im_nii):
    """
    Scale volumes and set the voxel size to 1mm x 1mm x 1mm
    If use subvolumes then create the subvolumes of the correct shape.
    Return the preprocessed volumes (scaling, isotropic resolution of 1mm) as well as the
    list of subvolumes for the fixed image, list of subvolumes for moving image and the coordinates of the subvolumes.
    """

    # Scale the data between 0 and 1
    fx_img = im_nii.get_fdata()
    scaled_fx_img = (fx_img - np.min(fx_img)) / (np.max(fx_img) - np.min(fx_img))

    mov_img = mov_im_nii.get_fdata()
    scaled_mov_img = (mov_img - np.min(mov_img)) / (np.max(mov_img) - np.min(mov_img))
    
    # Change the resolution to isotropic 1 mm resolution
    target_shape = np.array(scaled_fx_img.shape)
    new_resolution = [1, 1, 1]
    new_affine = np.zeros((4, 4))
    new_affine[:3, :3] = np.diag(new_resolution)
    # putting point 0,0,0 in the middle of the new volume - this could be refined in the future
    new_affine[:3, 3] = target_shape*new_resolution/2.*-1
    new_affine[3, 3] = 1.

    # Resample the to obtain the resolution wanted
    fx_resampled_nii = resample_img(nib.Nifti1Image(scaled_fx_img, im_nii.affine),
                                    target_affine=new_affine, interpolation='continuous')
    fx_img_res111 = fx_resampled_nii.get_fdata()
    mov_resampled_nii = resample_img(nib.Nifti1Image(scaled_mov_img, mov_im_nii.affine),
                                     target_affine=new_affine, interpolation='continuous')
    mov_img_res111 = mov_resampled_nii.get_fdata()

    fx_img_shape = fx_img_res111.shape
    mov_img_shape = mov_img_res111.shape

    max_img_shape = max(fx_img_shape, mov_img_shape)
    # Ensure that the volumes can be used in the registration model
    new_img_shape = (int(np.ceil(max_img_shape[0] // 16)) * 16, int(np.ceil(max_img_shape[1] // 16)) * 16,
                     int(np.ceil(max_img_shape[2] // 16)) * 16)

    # Pad the volumes to the max shape
    fx_img_pad_nii = resample_img(fx_resampled_nii, target_affine=new_affine,
                                  target_shape=new_img_shape, interpolation='continuous')
    fx_img_pad = fx_img_pad_nii.get_fdata()
    mov_img_pad_nii = resample_img(mov_resampled_nii, target_affine=new_affine,
                                   target_shape=new_img_shape, interpolation='continuous')
    mov_img_pad = mov_img_pad_nii.get_fdata()

    if data['use_subvol']:

        in_shape = (int(np.ceil(data['subvol_size'][0] // 16)) * 16, int(np.ceil(data['subvol_size'][1] // 16)) * 16,
                    int(np.ceil(data['subvol_size'][2] // 16)) * 16)

        # Determine how many subvolumes have to be created
        shape_in_vol = fx_img_pad.shape
        min_perc = 0.1

        nb_sub_x_axis = int(shape_in_vol[0] / (in_shape[0] - min_perc * in_shape[0])) + 1
        nb_sub_y_axis = int(shape_in_vol[1] / (in_shape[1] - min_perc * in_shape[1])) + 1
        nb_sub_z_axis = int(shape_in_vol[2] / (in_shape[2] - min_perc * in_shape[2])) + 1

        # Determine the number of overlapping voxels for each axis
        x_vox_overlap, y_vox_overlap, z_vox_overlap = 0, 0, 0
        if nb_sub_x_axis > 1:
            x_vox_overlap = (in_shape[0] - (shape_in_vol[0]/nb_sub_x_axis)) * (nb_sub_x_axis/(nb_sub_x_axis - 1))
        if nb_sub_y_axis > 1:
            y_vox_overlap = (in_shape[1] - (shape_in_vol[1]/nb_sub_y_axis)) * (nb_sub_y_axis/(nb_sub_y_axis - 1))
        if nb_sub_z_axis > 1:
            z_vox_overlap = (in_shape[2] - (shape_in_vol[2]/nb_sub_z_axis)) * (nb_sub_z_axis/(nb_sub_z_axis - 1))

        # Get the subvolumes and the coordinates of the x_min, x_max, y_min, y_max, z_min, z_max
        lst_subvol_fx = []
        lst_subvol_mov = []
        lst_coords_subvol = []

        x_max, y_max, z_max = 0, 0, 0
        for i in range(nb_sub_x_axis):
            x_min = 0 if i == 0 else int(x_max - x_vox_overlap)
            x_max = int(x_min + in_shape[0])
            for j in range(nb_sub_y_axis):
                y_min = 0 if j == 0 else int(y_max - y_vox_overlap)
                y_max = int(y_min + in_shape[1])
                for k in range(nb_sub_z_axis):
                    z_min = 0 if k == 0 else int(z_max - z_vox_overlap)
                    z_max = int(z_min + in_shape[2])
                    subvol_fx = fx_img_pad[x_min:x_max, y_min:y_max, z_min:z_max]
                    subvol_mov = mov_img_pad[x_min:x_max, y_min:y_max, z_min:z_max]

                    lst_subvol_fx.append(subvol_fx)
                    lst_subvol_mov.append(subvol_mov)
                    lst_coords_subvol.append((x_min, x_max, y_min, y_max, z_min, z_max))
    else:
        lst_subvol_fx, lst_subvol_mov, lst_coords_subvol = [], [], []

    # Create nifti files with the preprocessed volumes
    fx_preproc_nii = nib.Nifti1Image(fx_img_pad, new_affine)
    mov_preproc_nii = nib.Nifti1Image(mov_img_pad, new_affine)

    return fx_preproc_nii, mov_preproc_nii, lst_subvol_fx, lst_subvol_mov, lst_coords_subvol


def get_def_field_from_subvol(model_in_shape, im_shape, lst_coords_subvol, lst_warp_subvol):
    """
    Create a map of weights, apply it on the warping fields obtained with the different subvolumes,
    construct the final warping field and return it
    """
    # Create a map of weights that will be applied on the different warping fields to reduce the boundaries effect
    # via a weighted average of the overlapping volumes (between the volumes) giving more weights to the inside part
    x, y, z = model_in_shape[0]//2, model_in_shape[1]//2, model_in_shape[2]//2
    grid = np.mgrid[-x:x, -y:y, -z:z]
    w_map = np.maximum(np.abs(grid[0]), np.abs(grid[1]))
    w_map = np.maximum(w_map, np.abs(grid[2]))
    # The center of the volume has a weight of 1 and then it decreases linearly towards the boundaries
    w_map = 1 - w_map/(np.max(w_map) + 1)

    # Get the map representing the weights of all the subvolumes and place the map to the correct location of each
    # subvolume
    sum_weights = np.zeros((im_shape[0], im_shape[1], im_shape[2]))
    w_map_subvol_lst = []
    warp_subvol_lst = []
    for coords, warp in zip(lst_coords_subvol, lst_warp_subvol):
        w_map_subvol = np.zeros((im_shape[0], im_shape[1], im_shape[2]))
        warp_field_tmp = np.zeros((im_shape[0], im_shape[1], im_shape[2], 3))
        x_min, x_max, y_min, y_max, z_min, z_max = coords
        sum_weights[x_min:x_max, y_min:y_max, z_min:z_max] += w_map
        w_map_subvol[x_min:x_max, y_min:y_max, z_min:z_max] = w_map
        warp_field_tmp[x_min:x_max, y_min:y_max, z_min:z_max, :] = warp
        w_map_subvol_lst.append(w_map_subvol)
        warp_subvol_lst.append(warp_field_tmp)

    # To avoid division by 0, replace the sum weights that are 0 by 1 before the division (may appear for the voxels
    # at the border of the original volume if the size of this latter is even on certain axes)
    sum_weights[sum_weights == 0] = 1

    # Divide the weight map of each subvolume by the sum of all the weights of the different subvolumes to determine
    # the relative weight of each subvolume in the prediction of the final displacement vector
    w_map_subvol_final_lst = []
    for w_map_subvol in w_map_subvol_lst:
        w_map_subvol_final_lst.append(w_map_subvol/sum_weights)

    # Reconstruct the warping field
    warp_field = np.zeros((im_shape[0], im_shape[1], im_shape[2], 3))
    for w_subvol, warp in zip(w_map_subvol_final_lst, warp_subvol_lst):
        for i in range(3):
            warp_field[..., i] += w_subvol * warp[..., i]

    return warp_field


def register(data, reg_model, fx_im_path, mov_im_path, fx_contrast='T1w'):
    """
    Preprocess the two images and register the moving image to the fixed one using the provided model.
    Save the warped image and the deformation field.
    """

    fixed_nii = nib.load(f'{fx_im_path}.nii.gz')
    moving_nii = nib.load(f'{mov_im_path}.nii.gz')

    fixed, moving, lst_subvol_fx, lst_subvol_mov, lst_coords_subvol = \
        preprocess(data, fixed_nii, moving_nii)

    nib.save(fixed, os.path.join(f'{fx_im_path}_proc.nii.gz'))
    nib.save(moving, os.path.join(f'{mov_im_path}_proc.nii.gz'))

    if data['use_subvol']:
        model_in_shape = (int(np.ceil(data['subvol_size'][0] // 16)) * 16, int(np.ceil(data['subvol_size'][1] // 16)) * 16,
                          int(np.ceil(data['subvol_size'][2] // 16)) * 16)
    else:
        model_in_shape = fixed.get_fdata().shape

    reg_args = dict(
        inshape=model_in_shape,
        int_steps=data['int_steps'],
        int_resolution=data['int_res'],
        svf_resolution=data['svf_res'],
        nb_unet_features=(data['enc'], data['dec'])
    )

    model = vxm.networks.VxmDense(**reg_args)
    model.set_weights(reg_model.get_weights())

    if not data['use_subvol']:
        moved, warp = model.predict([np.expand_dims(moving.get_fdata().squeeze(), axis=(0, -1)),
                                     np.expand_dims(fixed.get_fdata().squeeze(), axis=(0, -1))])
        warp_data = warp[0, ...]
        is_warp_half_res = False if warp_data.shape[0] == model_in_shape[0] else True
        if is_warp_half_res:
            warp_data = zoom(warp_data, (2, 2, 2, 1))
        warp = nib.Nifti1Image(warp_data, fixed.affine)
        moved_nii = nib.Nifti1Image(moved[0, ..., 0], fixed.affine)
        nib.save(moved_nii, os.path.join(f'{mov_im_path}_proc_reg_to_{fx_contrast}.nii.gz'))
        nib.save(warp, os.path.join(f'{mov_im_path}_proc_field_to_{fx_contrast}.nii.gz'))
        warp_in_original_space = resample_img(warp, target_affine=moving_nii.affine,
                                              target_shape=moving_nii.get_fdata().shape, interpolation='continuous')
        nib.save(warp_in_original_space, os.path.join(f'{mov_im_path}_warp_original_dim.nii.gz'))
    else:
        warp_field_lst = []
        for fx_subvol, mov_subvol in zip(lst_subvol_fx, lst_subvol_mov):
            _, warp = model.predict([np.expand_dims(mov_subvol.squeeze(), axis=(0, -1)),
                                     np.expand_dims(fx_subvol.squeeze(), axis=(0, -1))])
            warp_field_lst.append(warp[0, ...])

        is_warp_half_res = False if warp_field_lst[0].shape[0] == model_in_shape[0] else True

        if is_warp_half_res:
            warp_field_lst_good_dim = []
            for warp in warp_field_lst:
                warp_field_lst_good_dim.append(zoom(warp, (2, 2, 2, 1)))
        else:
            warp_field_lst_good_dim = warp_field_lst

        warp_field = get_def_field_from_subvol(model_in_shape, moving.shape, lst_coords_subvol, warp_field_lst_good_dim)

        def_field_nii = nib.Nifti1Image(warp_field, affine=fixed.affine)
        nib.save(def_field_nii, os.path.join(f'{mov_im_path}_proc_field_to_{fx_contrast}.nii.gz'))
        warp_in_original_space = resample_img(def_field_nii, target_affine=moving_nii.affine,
                                              target_shape=moving_nii.get_fdata().shape, interpolation='continuous')
        nib.save(warp_in_original_space, os.path.join(f'{mov_im_path}_warp_original_dim.nii.gz'))

        moving = vxm.py.utils.load_volfile(os.path.join(f'{mov_im_path}_proc.nii.gz'),
                                           add_batch_axis=True, add_feat_axis=True)
        warp_to_apply = vxm.py.utils.load_volfile(os.path.join(f'{mov_im_path}_proc_field_to_{fx_contrast}.nii.gz'),
                                                  add_batch_axis=True, ret_affine=True)

        moved = vxm.networks.Transform(moving.shape[1:-1], nb_feats=moving.shape[-1]).predict([moving, warp_to_apply[0]])
        # save moved image
        vxm.py.utils.save_volfile(moved.squeeze(), os.path.join(f'{mov_im_path}_proc_reg_to_{fx_contrast}.nii.gz'), fixed.affine)

    moved_data = moved.squeeze()
    moved_nii = nib.Nifti1Image(moved_data, fixed.affine)
    moved_in_original_space = resample_img(moved_nii, target_affine=moving_nii.affine,
                                           target_shape=moving_nii.get_fdata().shape, interpolation='continuous')
    nib.save(moved_in_original_space, os.path.join(f'{mov_im_path}_reg_original_dim.nii.gz'))


def run_main(data, reg_model_path, fx_im_path, mov_im_path, fx_im_contrast='T1w'):
    """
    Load the registration model
    Preprocess the fixed and moving images
    Register
    """
    # Load the registration model
    model = vxm.networks.VxmDense.load(reg_model_path, input_model=None)

    register(data, model, fx_im_path, mov_im_path, fx_contrast=fx_im_contrast)


if __name__ == "__main__":

    # parse the commandline
    parser = argparse.ArgumentParser()

    # parameters to be specified by the user
    parser.add_argument('--model-path', required=True, type=str, help='path to the registration model')

    parser.add_argument('--fx-img-path', required=True, help='path to the fixed image')
    parser.add_argument('--mov-img-path', required=True, help='path to the moving image')

    parser.add_argument('--fx-img-contrast', required=False, default='T1w',
                        help='contrast of the fixed image: one of {T1w, T2w, T2star}')

    parser.add_argument('--one-cpu-tf', required=False, type=str, default='True',
                        help='boolean to determine if the processes link to TF have access to one CPU (True) or all '
                             'the CPUs (False) {\'0\',\'1\', \'False\',\'True\'}')

    args = parser.parse_args()

    # ***************************************************************************************
    # TODO - Add a config file where the parameters are specified
    # import json
    # with open(args.config_path) as config_file:
    #     data = json.load(config_file)

    data = dict(
        use_subvol=True,
        subvol_size=[160, 160, 192],
        int_steps=5,
        int_res=2,
        svf_res=2,
        enc=[256, 256, 256, 256],
        dec=[256, 256, 256, 256, 256, 256]
    )
    # ***************************************************************************************

    if eval(args.one_cpu_tf):
        # set that TF can use only one CPU
        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess = tf.compat.v1.Session(config=session_conf)

    run_main(data, args.model_path, args.fx_img_path, args.mov_img_path, args.fx_img_contrast)
