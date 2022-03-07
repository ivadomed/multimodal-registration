"""
File to select the moving image and preprocess the fixed and moving pair
"""

import os
import argparse

import json
import numpy as np
import nibabel as nib
import pandas as pd
import csv
import datetime

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

    resample_interp = model_inference_specs['resample_interpolation']
    if resample_interp not in ['nearest', 'linear', 'spline']:
        resample_interp = 'linear'
    if resample_interp == 'nearest':
        resample_interp = 'nn'

    # Scale the data between 0 and 1
    fx_img = im_nii.get_fdata()
    scaled_fx_img = (fx_img - np.min(fx_img)) / (np.max(fx_img) - np.min(fx_img))

    mov_img = mov_im_nii.get_fdata()
    scaled_mov_img = (mov_img - np.min(mov_img)) / (np.max(mov_img) - np.min(mov_img))

    # Change the resolution to isotropic 1 mm resolution
    fx_resampled_nii = resample_nib(nib.Nifti1Image(scaled_fx_img, im_nii.affine), new_size=[1, 1, 1],
                                    new_size_type='mm', image_dest=None, interpolation=resample_interp, mode='constant')
    fx_img_res111 = fx_resampled_nii.get_fdata()
    mov_resampled_nii = resample_nib(nib.Nifti1Image(scaled_mov_img, mov_im_nii.affine),
                                     image_dest=fx_resampled_nii, interpolation=resample_interp, mode='constant')
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


def select_random_sub(data_path, fx_im_path, mult_sessions, mov_contrast, out_file='../../reg_pairs.csv', append=1):
    """
    Select a random subject of the dataset that is different than the one of the fixed image.
    And write in a csv file the pair of subjects used (for reproducibility)
    """

    fixed_sub = (fx_im_path.split('_')[0]).split('/')[-1]

    sub_lst = [sub for sub in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, sub))]
    sub_lst.remove(fixed_sub)
    moving_sub = np.random.choice(sub_lst)
    mov_im_path = os.path.join(data_path, moving_sub, 'anat', f'{moving_sub}_{mov_contrast}.nii.gz')

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

    return mov_im_path


def run_main(model_inference_specs, fx_im_path, data_path, mult_sessions=False,
             mov_contrast='T1w', inter_sub_csv=None):
    """
    Select a moving image
    Preprocess the fixed and moving images
    """

    if inter_sub_csv is None:
        mov_im_path = select_random_sub(data_path, fx_im_path, mult_sessions, mov_contrast)
    else:
        df = pd.read_csv(inter_sub_csv)
        fixed_sub = (fx_im_path.split('_')[0]).split('/')[-1]
        mov_subject = df.loc[df['fixed_subject'] == fixed_sub]['moving_subject']
        mov_im_path = os.path.join(data_path, mov_subject, 'anat', f'{mov_subject}_{mov_contrast}.nii.gz')

    fixed_nii = nib.load(fx_im_path)
    moving_nii = nib.load(mov_im_path)
    nib.save(moving_nii, os.path.join(f'moving_before_proc.nii.gz'))

    fx_im_path = fx_im_path.split(".")[0]
    # TODO - define a move im path that has the information of the fixed and moving images used for the registration
    mov_im_path = 'moving'

    fixed, moving, lst_subvol_fx, lst_subvol_mov, lst_coords_subvol = \
        preprocess(model_inference_specs, fixed_nii, moving_nii)

    nib.save(fixed, os.path.join(f'{fx_im_path}_proc.nii.gz'))
    nib.save(moving, os.path.join(f'{mov_im_path}_proc.nii.gz'))

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
