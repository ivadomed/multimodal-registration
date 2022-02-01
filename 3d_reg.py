"""
File to load a trained registration model and register two images together.
It includes a preprocessing step to scale the volumes and set it to an isotropic resolution of 1mm as required by the model.
"""

import os
import argparse

import json
import numpy as np
import nibabel as nib
import voxelmorph as vxm

from nilearn.image import resample_img
from nibabel.processing import resample_from_to


def resample_nib(image, new_size=None, new_size_type=None, image_dest=None, interpolation='linear', mode='nearest'):
    """
    Resample a nibabel or Image object based on a specified resampling factor.
    Can deal with 2d, 3d or 4d image objects.
    Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
    Authors: Julien Cohen-Adad, Sara Dupont

    :param image: nibabel or Image image.
    :param new_size: list of float: Resampling factor, final dimension or resolution, depending on new_size_type.
    :param new_size_type: {'vox', 'factor', 'mm'}: Feature used for resampling. Examples:
        new_size=[128, 128, 90], new_size_type='vox' --> Resampling to a dimension of 128x128x90 voxels
        new_size=[2, 2, 2], new_size_type='factor' --> 2x isotropic upsampling
        new_size=[1, 1, 5], new_size_type='mm' --> Resampling to a resolution of 1x1x5 mm
    :param image_dest: Destination image to resample the input image to. In this case, new_size and new_size_type
        are ignored
    :param interpolation: {'nn', 'linear', 'spline'}. The interpolation type
    :param mode: Outside values are filled with 0 ('constant') or nearest value ('nearest').
    :return: The resampled nibabel or Image image (depending on the input object type).
    """

    # set interpolation method
    dict_interp = {'nn': 0, 'linear': 1, 'spline': 2}

    # If input is an Image object, create nibabel object from it
    if type(image) == nib.nifti1.Nifti1Image:
        img = image
    else:
        raise Exception(TypeError)

    if image_dest is None:
        # Get dimensions of data
        p = img.header.get_zooms()
        shape = img.header.get_data_shape()

        if img.ndim == 4:
            new_size += ['1']  # needed because the code below is general, i.e., does not assume 3d input and uses img.shape

        # compute new shape based on specific resampling method
        if new_size_type == 'vox':
            shape_r = tuple([int(new_size[i]) for i in range(img.ndim)])
        elif new_size_type == 'factor':
            if len(new_size) == 1:
                # isotropic resampling
                new_size = tuple([new_size[0] for i in range(img.ndim)])
            # compute new shape as: shape_r = shape * f
            shape_r = tuple([int(np.round(shape[i] * float(new_size[i]))) for i in range(img.ndim)])
        elif new_size_type == 'mm':
            if len(new_size) == 1:
                # isotropic resampling
                new_size = tuple([new_size[0] for i in range(img.ndim)])
            # compute new shape as: shape_r = shape * (p_r / p)
            shape_r = tuple([int(np.round(shape[i] * float(p[i]) / float(new_size[i]))) for i in range(img.ndim)])
        else:
            raise ValueError("'new_size_type' is not recognized.")

        # Generate 3d affine transformation: R
        affine = img.affine[:4, :4]
        affine[3, :] = np.array([0, 0, 0, 1])  # satisfy to nifti convention. Otherwise it grabs the temporal
        # logger.debug('Affine matrix: \n' + str(affine))
        R = np.eye(4)
        for i in range(3):
            try:
                R[i, i] = img.shape[i] / float(shape_r[i])
            except ZeroDivisionError:
                raise ZeroDivisionError("Destination size is zero for dimension {}. You are trying to resample to an "
                                        "unrealistic dimension. Check your NIFTI pixdim values to make sure they are "
                                        "not corrupted.".format(i))

        affine_r = np.dot(affine, R)
        reference = (shape_r, affine_r)

    # If reference is provided
    else:
        if type(image_dest) == nib.nifti1.Nifti1Image:
            reference = image_dest
        else:
            raise Exception(TypeError)

    if img.ndim == 3:
        # we use mode 'nearest' to overcome issue #2453
        img_r = resample_from_to(img, to_vox_map=reference, order=dict_interp[interpolation],
                                 mode=mode, cval=0.0, out_class=None)

    elif img.ndim == 4:
        # Import here instead of top of the file because this is an isolated case and nibabel takes time to import
        data4d = np.zeros(shape_r)
        # Loop across 4th dimension and resample each 3d volume
        for it in range(img.shape[3]):
            # Create dummy 3d nibabel image
            nii_tmp = nib.nifti1.Nifti1Image(img.get_data()[..., it], affine)
            img3d_r = resample_from_to(nii_tmp, to_vox_map=(shape_r[:-1], affine_r),
                                       order=dict_interp[interpolation], mode=mode, cval=0.0, out_class=None)
            data4d[..., it] = img3d_r.get_data()
        # Create 4d nibabel Image
        img_r = nib.nifti1.Nifti1Image(data4d, affine_r)
        # Copy over the TR parameter from original 4D image (otherwise it will be incorrectly set to 1)
        img_r.header.set_zooms(list(img_r.header.get_zooms()[0:3]) + [img.header.get_zooms()[3]])

    return img_r


def preprocess(model_inference_specs, im_nii, mov_im_nii):
    """
    Scale volumes and create subvolumes of the correct shape.
    Return the preprocessed volumes (scaling, zero-padding, isotropic resolution of 1mm) as well as the
    list of subvolumes for the fixed image, list of subvolumes for moving image and the coordinates of the subvolumes.
    """

    # Scale the data between 0 and 1
    fx_img = im_nii.get_fdata()
    scaled_fx_img = (fx_img - np.min(fx_img)) / (np.max(fx_img) - np.min(fx_img))

    mov_img = mov_im_nii.get_fdata()
    scaled_mov_img = (mov_img - np.min(mov_img)) / (np.max(mov_img) - np.min(mov_img))

    # Change the resolution to isotropic 1 mm resolution
    fx_resampled_nii = resample_nib(nib.Nifti1Image(scaled_fx_img, im_nii.affine), new_size=[1, 1, 1],
                                    new_size_type='mm', image_dest=None, interpolation='linear', mode='constant')
    fx_img_res111 = fx_resampled_nii.get_fdata()
    mov_resampled_nii = resample_nib(nib.Nifti1Image(scaled_mov_img, mov_im_nii.affine),
                                     image_dest=fx_resampled_nii, interpolation='linear', mode='constant')
    mov_img_res111 = mov_resampled_nii.get_fdata()

    # Ensure that the volumes can be used in the registration model
    fx_img_shape = fx_img_res111.shape
    mov_img_shape = mov_img_res111.shape
    max_img_shape = max(fx_img_shape, mov_img_shape)
    new_img_shape = (int(np.ceil(max_img_shape[0] // 16)) * 16, int(np.ceil(max_img_shape[1] // 16)) * 16,
                     int(np.ceil(max_img_shape[2] // 16)) * 16)

    # Pad the volumes to the max shape
    fx_resampled_nii = resample_img(fx_resampled_nii, target_affine=fx_resampled_nii.affine,
                                    target_shape=new_img_shape, interpolation='continuous')
    fx_img_res111 = fx_resampled_nii.get_fdata()
    mov_resampled_nii = resample_img(mov_resampled_nii, target_affine=mov_resampled_nii.affine,
                                     target_shape=new_img_shape, interpolation='continuous')
    mov_img_res111 = mov_resampled_nii.get_fdata()

    if model_inference_specs['use_subvol']:

        in_shape = (int(np.ceil(model_inference_specs['subvol_size'][0] // 16)) * 16,
                    int(np.ceil(model_inference_specs['subvol_size'][1] // 16)) * 16,
                    int(np.ceil(model_inference_specs['subvol_size'][2] // 16)) * 16)

        # Determine how many subvolumes have to be created
        shape_in_vol = fx_img_res111.shape
        min_perc = model_inference_specs['min_perc_overlap']
        if min_perc >= 1:
            if min_perc/100 < 1:
                min_perc = min_perc/100
            else:
                min_perc = 0.1
        elif min_perc <= 0:
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
                    subvol_fx = fx_img_res111[x_min:x_max, y_min:y_max, z_min:z_max]
                    subvol_mov = mov_img_res111[x_min:x_max, y_min:y_max, z_min:z_max]

                    lst_subvol_fx.append(subvol_fx)
                    lst_subvol_mov.append(subvol_mov)
                    lst_coords_subvol.append((x_min, x_max, y_min, y_max, z_min, z_max))
    else:
        lst_subvol_fx, lst_subvol_mov, lst_coords_subvol = [], [], []

    return fx_resampled_nii, mov_resampled_nii, lst_subvol_fx, lst_subvol_mov, lst_coords_subvol


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


def run_main(model_inference_specs, model_path, fx_im_path, mov_im_path, res_dir='res',
             warp_interp='linear', out_im_path='warped_im', out_field_path='deform_field'):
    """
    Load a registration model, preprocess the two images and register the moving image to the fixed one.
    Save the warped image and the deformation field in the paths specified.
    """

    if warp_interp not in ['nearest', 'linear']:
        warp_interp = 'linear'
    
    model = vxm.networks.VxmDense.load(model_path, input_model=None)
    reg_model = model

    os.makedirs(res_dir, exist_ok=True)
    moved_path = os.path.join(res_dir, f'{out_im_path}.nii.gz')
    warp_path = os.path.join(res_dir, f'{out_field_path}.nii.gz')

    fixed_nii = nib.load(fx_im_path)
    moving_nii = nib.load(mov_im_path)

    fixed, moving, lst_subvol_fx, lst_subvol_mov, lst_coords_subvol = \
        preprocess(model_inference_specs, fixed_nii, moving_nii)

    if model_inference_specs['use_subvol']:
        model_in_shape = (int(np.ceil(model_inference_specs['subvol_size'][0] // 16)) * 16,
                          int(np.ceil(model_inference_specs['subvol_size'][1] // 16)) * 16,
                          int(np.ceil(model_inference_specs['subvol_size'][2] // 16)) * 16)
    else:
        model_in_shape = fixed.get_fdata().shape

    reg_args = dict(
        inshape=model_in_shape,
        int_steps=model_inference_specs['int_steps'],
        int_resolution=model_inference_specs['int_res'],
        svf_resolution=model_inference_specs['svf_res'],
        nb_unet_features=(model_inference_specs['enc'], model_inference_specs['dec'])
    )

    model = vxm.networks.VxmDense(**reg_args)
    model.set_weights(reg_model.get_weights())

    if not model_inference_specs['use_subvol']:
        _, warp = model.predict([np.expand_dims(moving.get_fdata().squeeze(), axis=(0, -1)),
                                 np.expand_dims(fixed.get_fdata().squeeze(), axis=(0, -1))])
        warp_data = warp[0, ...]

        is_warp_half_res = False if warp_data.shape[0] == model_in_shape[0] else True
        if is_warp_half_res:
            warp_affine = np.copy(fixed.affine)
            for i in range(3):
                warp_affine[i, i] = warp_affine[i, i] * 2
            warp = nib.Nifti1Image(warp_data, warp_affine)
            warp = resample_nib(warp, new_size=[2, 2, 2, 1], new_size_type='factor', image_dest=None,
                                interpolation='linear', mode='constant')
        else:
            warp = nib.Nifti1Image(warp_data, fixed.affine)

        nib.save(warp, os.path.join(f'{mov_im_path}_proc_field.nii.gz'))
        warp_in_original_space = resample_img(warp, target_affine=moving_nii.affine,
                                              target_shape=moving_nii.get_fdata().shape, interpolation='continuous')
        nib.save(warp_in_original_space, warp_path)

        nib.save(moving, os.path.join(f'{mov_im_path}_proc.nii.gz'))
        moving = vxm.py.utils.load_volfile(os.path.join(f'{mov_im_path}_proc.nii.gz'),
                                           add_batch_axis=True, add_feat_axis=True)
        warp_to_apply = vxm.py.utils.load_volfile(os.path.join(f'{mov_im_path}_proc_field.nii.gz'),
                                                  add_batch_axis=True, ret_affine=True)

        os.remove(os.path.join(f'{mov_im_path}_proc.nii.gz'))
        os.remove(os.path.join(f'{mov_im_path}_proc_field.nii.gz'))

        moved = vxm.networks.Transform(moving.shape[1:-1],
                                       interp_method=warp_interp,
                                       nb_feats=moving.shape[-1]).predict([moving, warp_to_apply[0]])

    else:
        
        warp_field_lst = []
        for fx_subvol, mov_subvol in zip(lst_subvol_fx, lst_subvol_mov):
            _, warp = model.predict([np.expand_dims(mov_subvol.squeeze(), axis=(0, -1)),
                                     np.expand_dims(fx_subvol.squeeze(), axis=(0, -1))])
            warp_field_lst.append(warp[0, ...])

        is_warp_half_res = False if warp_field_lst[0].shape[0] == model_in_shape[0] else True

        if is_warp_half_res:
            model_in_shape = np.array(model_in_shape)
            moving_shape = np.array(moving.shape)
            for i in range(3):
                model_in_shape[i] = model_in_shape[i] // 2
                moving_shape[i] = moving_shape[i] // 2
            new_coords = []
            for coord in lst_coords_subvol:
                x_min, x_max, y_min, y_max, z_min, z_max = coord
                x_min, x_max, y_min, y_max, z_min, z_max = x_min // 2, x_max // 2, y_min // 2, y_max // 2, z_min // 2, z_max // 2
                new_coords.append((x_min, x_max, y_min, y_max, z_min, z_max))
            lst_coords_subvol = new_coords
        else:
            moving_shape = moving.shape

        warp_field = get_def_field_from_subvol(model_in_shape, moving_shape, lst_coords_subvol, warp_field_lst)

        if is_warp_half_res:
            warp_affine = np.copy(fixed.affine)
            for i in range(3):
                warp_affine[i, i] = warp_affine[i, i] * 2
            warp = nib.Nifti1Image(warp_field, warp_affine)
            def_field_nii = resample_nib(warp, new_size=[2, 2, 2, 1], new_size_type='factor', image_dest=None,
                                         interpolation='spline', mode='constant')
        else:
            def_field_nii = nib.Nifti1Image(warp_field, affine=fixed.affine)

        nib.save(def_field_nii, os.path.join(f'{mov_im_path}_proc_field.nii.gz'))
        warp_in_original_space = resample_img(def_field_nii, target_affine=moving_nii.affine,
                                              target_shape=moving_nii.get_fdata().shape, interpolation='continuous')
        nib.save(warp_in_original_space, warp_path)

        nib.save(moving, os.path.join(f'{mov_im_path}_proc.nii.gz'))
        moving = vxm.py.utils.load_volfile(os.path.join(f'{mov_im_path}_proc.nii.gz'),
                                           add_batch_axis=True, add_feat_axis=True)
        warp_to_apply = vxm.py.utils.load_volfile(os.path.join(f'{mov_im_path}_proc_field.nii.gz'),
                                                  add_batch_axis=True, ret_affine=True)
        
        os.remove(os.path.join(f'{mov_im_path}_proc.nii.gz'))
        os.remove(os.path.join(f'{mov_im_path}_proc_field.nii.gz'))

        moved = vxm.networks.Transform(moving.shape[1:-1],
                                       interp_method=warp_interp,
                                       nb_feats=moving.shape[-1]).predict([moving, warp_to_apply[0]])

    moved_data = moved.squeeze()
    moved_nii = nib.Nifti1Image(moved_data, fixed.affine)
    moved_in_original_space = resample_img(moved_nii, target_affine=moving_nii.affine,
                                           target_shape=moving_nii.get_fdata().shape, interpolation='continuous')

    nib.save(moved_in_original_space, moved_path)


if __name__ == "__main__":

    # parse the commandline
    parser = argparse.ArgumentParser()

    # parameters to be specified by the user
    parser.add_argument('--model-path', required=True, type=str, help='path to the registration model')
    parser.add_argument('--config-path', required=True, type=str,
                        help='path to the config file with the inference model specificities')

    parser.add_argument('--fx-img-path', required=True, help='path to the fixed image')
    parser.add_argument('--mov-img-path', required=True, help='path to the moving image')

    parser.add_argument('--res-dir', required=False, default='res', help='results output directory (default: res)')

    parser.add_argument('--warp-interp', default='linear',
                        help='interpolation method to obtain the registered volume using the warping field outputted '
                             'by the registration model. Choice between linear and nearest (default: linear)')
    
    parser.add_argument('--out-img-name', required=False, default='warped_im',
                        help='name of the warped image that will result')
    parser.add_argument('--def-field-name', required=False, default='deform_field',
                        help='name of the deformation field that will result')

    args = parser.parse_args()

    with open(args.config_path) as config_file:
        model_inference_specs = json.load(config_file)

    run_main(model_inference_specs, args.model_path, args.fx_img_path, args.mov_img_path, 
             args.res_dir, args.warp_interp, args.out_img_name, args.def_field_name)
