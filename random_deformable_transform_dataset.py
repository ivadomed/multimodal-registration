"""
File to randomly transform the volumes using warping field generated from noise distribution
"""

import os
import argparse

import numpy as np
import nibabel as nib
import neurite as ne
import voxelmorph as vxm
import tensorflow.keras.backend as K

import csv
import datetime

if __name__ == "__main__":

    # parse the commandline
    parser = argparse.ArgumentParser()

    # parameters to be specified by the user
    parser.add_argument('--mov-img-path', required=True, help='path to the moving image')

    parser.add_argument('--sub-id', required=True, help='Subject ID')
    parser.add_argument('--out-file', required=False, default='summary_transform.csv',
                        help='path to the output csv summarizing the affine transform applied')

    args = parser.parse_args()

    # -------------------------------------------------------------------------------------------------------- #
    # ----                  LOADING THE VOLUME AND GETTING THE ASSOCIATED AFFINE MATRIX                   ---- #
    # -------------------------------------------------------------------------------------------------------- #

    im = nib.load(f"{args.mov_img_path}.nii.gz")
    affine = im.affine

    # -------------------------------------------------------------------------------------------------------- #
    # ----                                GENERATING THE DEFORMATION FIELD                                ---- #
    # -------------------------------------------------------------------------------------------------------- #

    def_field, std = ne.utils.augment.draw_perlin(out_shape=(im.shape[0], im.shape[1], im.shape[2], 1, 3),
                                                  scales=[16], max_std=2.5)

    def_field2, std2 = ne.utils.augment.draw_perlin(out_shape=(im.shape[0], im.shape[1], im.shape[2], 1, 3),
                                                    scales=[32, 64], max_std=5)

    warp = vxm.utils.compose([K.constant(def_field[..., 0, :]), K.constant(def_field2[..., 0, :])])
    warp_data = K.eval(warp)
    def_field_nii = nib.Nifti1Image(np.array(warp_data), affine=affine)

    out_def_path = f"{args.mov_img_path}_warp_to_transform.nii.gz"
    nib.save(def_field_nii, out_def_path)

    # -------------------------------------------------------------------------------------------------------- #
    # ----             APPLYING THE DEFORMATION FIELD TO THE IMAGE TO PRODUCE THE MOVED IMAGE             ---- #
    # -------------------------------------------------------------------------------------------------------- #

    moving = vxm.py.utils.load_volfile(f"{args.mov_img_path}.nii.gz", add_batch_axis=True, add_feat_axis=True)
    deform = vxm.py.utils.load_volfile(out_def_path, add_batch_axis=True, ret_affine=True)

    moved = vxm.networks.Transform(moving.shape[1:-1],
                                   interp_method='linear',
                                   nb_feats=moving.shape[-1]).predict([moving, deform[0]])

    # save moved image
    out_im_path = f"{args.mov_img_path}_def_transformed.nii.gz"
    vxm.py.utils.save_volfile(moved.squeeze(), out_im_path, affine)

    # os.remove(out_def_path)

    summary_transfo = dict()
    summary_transfo['subject'] = args.sub_id
    summary_transfo['std_for_scale_16'] = std[0]
    summary_transfo['std_for_scale_32'] = std2[0]
    summary_transfo['std_for_scale_64'] = std2[1]

    # write header
    if not os.path.isfile(args.out_file):
        with open(args.out_file, 'w') as csvfile:
            header = ['Timestamp', 'Subject', 'std_for_scale_16', 'std_for_scale_32', 'std_for_scale_64']
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()

    # populate data
    with open(args.out_file, 'a') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        line = list()
        line.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))  # Timestamp
        for val in summary_transfo.keys():
            line.append(str(summary_transfo[val]))
        spamwriter.writerow(line)
