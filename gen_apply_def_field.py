"""
This file takes as input a volume and some parameters to generate a deformation field.
It then generates a deformation field based on the specified parameters following the
Perlin noise strategy used in synthmorph.
This deformation field is then applied to the input volume to obtain a moved object.
The moved volume as well as the deformation field generated are saved to the paths specified.
"""

import argparse

import numpy as np
import nibabel as nib

import neurite as ne
import voxelmorph as vxm


if __name__ == "__main__":

    # -------------------------------------------------------------------------------------------------------- #
    # ----                                       PARSER ARGUMENTS                                         ---- #
    # -------------------------------------------------------------------------------------------------------- #

    # parse command line
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=f'Deform an image with the generated deformation field')

    # path parameters
    p.add_argument('--im-path', required=True, help='path to the volume to deform')
    p.add_argument('--out-im-path', default='moved_im.nii.gz', help='path where the moved volume will be saved')
    p.add_argument('--out-def-path', default='deformation_field.nii.gz', help='path where the def. field will be saved')

    # generation deformation field parameters
    p.add_argument('--def-scales', type=int, nargs='+', default=[16, 32, 64],
                   help='list of relative resolutions at which noise is sampled normally (default: 16 32 64)')
    p.add_argument('--def-max-std', type=int, default=3,
                   help='max std for the gaussian dist of noise in label maps generation (def field) (default: 16)')

    # application of deformation field
    p.add_argument('--interp', default='linear', help='interpolation method linear/nearest (default: linear)')

    arg = p.parse_args()

    # -------------------------------------------------------------------------------------------------------- #
    # ----                  LOADING THE VOLUME AND GETTING THE ASSOCIATED AFFINE MATRIX                   ---- #
    # -------------------------------------------------------------------------------------------------------- #

    im = nib.load(arg.im_path)
    affine = im.affine

    # -------------------------------------------------------------------------------------------------------- #
    # ----                                GENERATING THE DEFORMATION FIELD                                ---- #
    # -------------------------------------------------------------------------------------------------------- #

    def_field = ne.utils.augment.draw_perlin(out_shape=(im.shape[0], im.shape[1], im.shape[2], 1, 3),
                                             scales=arg.def_scales, max_std=arg.def_max_std)

    def_field_nii = nib.Nifti1Image(np.array(def_field[..., 0, :]), affine=affine)
    nib.save(def_field_nii, arg.out_def_path)

    # -------------------------------------------------------------------------------------------------------- #
    # ----             APPLYING THE DEFORMATION FIELD TO THE IMAGE TO PRODUCE THE MOVED IMAGE             ---- #
    # -------------------------------------------------------------------------------------------------------- #

    moving = vxm.py.utils.load_volfile(arg.im_path, add_batch_axis=True, add_feat_axis=True)
    deform = vxm.py.utils.load_volfile(arg.out_def_path, add_batch_axis=True, ret_affine=True)

    # moved = vxm.networks.Transform(im.shape, interp_method=arg.interp, nb_feats=1).predict([im, def_field])
    moved = vxm.networks.Transform(moving.shape[1:-1],
                                   interp_method=arg.interp,
                                   nb_feats=moving.shape[-1]).predict([moving, deform[0]])

    # save moved image
    vxm.py.utils.save_volfile(moved.squeeze(), arg.out_im_path, affine)
