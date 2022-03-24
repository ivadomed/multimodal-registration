"""
File to randomly affine transform the volumes
"""

import os
import argparse

import numpy as np
import nibabel as nib

from scipy.ndimage.interpolation import affine_transform
import csv
import datetime


def random_affine_transform(im, sub_id, out_file):

    import math
    import random

    angle_degree = 5  # 5
    # Get the random angle
    angle_d = np.random.uniform(- angle_degree, angle_degree)
    angle = math.radians(angle_d)
    # Get the two axes that define the plane of rotation
    axes = list(random.sample(range(3), 2))
    axes.sort()

    scale_factor = 0.05  # 0.05
    # Scale
    scale_axis = random.uniform(1 - scale_factor, 1 + scale_factor)

    # Get params
    data_shape = im.shape
    translation = [0.05, 0.05, 0.05]  # 0.05, 0.05, 0.05
    max_dx = translation[0] * data_shape[0]
    max_dy = translation[1] * data_shape[1]
    max_dz = translation[2] * data_shape[2]
    translations = (np.round(np.random.uniform(-max_dx, max_dx)),
                    np.round(np.random.uniform(-max_dy, max_dy)),
                    np.round(np.random.uniform(-max_dz, max_dz)))

    # Do rotation
    shape = 0.5 * np.array(data_shape)
    if axes == [0, 1]:
        rotate = np.array([[math.cos(angle), -math.sin(angle), 0],
                           [math.sin(angle), math.cos(angle), 0],
                           [0, 0, 1]])
    elif axes == [0, 2]:
        rotate = np.array([[math.cos(angle), 0, math.sin(angle)],
                           [0, 1, 0],
                           [-math.sin(angle), 0, math.cos(angle)]])
    elif axes == [1, 2]:
        rotate = np.array([[1, 0, 0],
                           [0, math.cos(angle), -math.sin(angle)],
                           [0, math.sin(angle), math.cos(angle)]])
    else:
        raise ValueError("Unknown axes value")

    scale = np.array([[1 / scale_axis, 0, 0], [0, 1 / scale_axis, 0], [0, 0, 1 / scale_axis]])
    transforms = scale.dot(rotate)

    offset = shape - shape.dot(transforms) + translations

    data_out = affine_transform(im, transforms.T, order=1, offset=offset,
                                output_shape=data_shape).astype(im.dtype)

    summary_transfo = dict()
    summary_transfo['subject'] = sub_id
    summary_transfo['rotation_angle_degree'] = angle_d
    summary_transfo['rotation_axes'] = axes
    summary_transfo['scaling'] = scale_axis
    summary_transfo['translation'] = translations
    summary_transfo['im_shape'] = im.shape

    # write header
    if not os.path.isfile(out_file):
        with open(out_file, 'w') as csvfile:
            header = ['Timestamp', 'Subject', 'rotation_angle_degree', 'rotation_axes', 'scaling', 'translation', 'im_shape']
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()

    # populate data
    with open(out_file, 'a') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        line = list()
        line.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))  # Timestamp
        for val in summary_transfo.keys():
            line.append(str(summary_transfo[val]))
        spamwriter.writerow(line)

    return data_out


def run_main(mov_im_path, sub_id, out_file):
    """
    Transform the moving volume
    """

    moving_nii = nib.load(f'{mov_im_path}.nii.gz')

    mov_data = moving_nii.get_fdata()
    affine_transform_mov_data = random_affine_transform(mov_data, sub_id, out_file)
    mov_resampled_nii = nib.Nifti1Image(affine_transform_mov_data, moving_nii.affine)

    nib.save(mov_resampled_nii, os.path.join(f'{mov_im_path}_aff_transformed.nii.gz'))


if __name__ == "__main__":

    # parse the commandline
    parser = argparse.ArgumentParser()

    # parameters to be specified by the user
    parser.add_argument('--mov-img-path', required=True, help='path to the moving image')

    parser.add_argument('--sub-id', required=True, help='Subject ID')
    parser.add_argument('--out-file', required=False, default='summary_transform.csv',
                        help='path to the output csv summarizing the affine transform applied')

    args = parser.parse_args()

    run_main(args.mov_img_path, args.sub_id, args.out_file)

