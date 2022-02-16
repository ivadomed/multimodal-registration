"""
File taking the dense deformation field as input and computing the percentage of voxels with a negative Jacobian
determinant, representative of local folding.
A Jacobian determinant higher than 1.0 represents voxel expansion; a Jacobian determinant lower than 1.0 represents voxel compression.
The different values associated to the Jacobian and its determinant are saved in a csv file.
A volume with voxel intensities representative of the determinant of the Jacobian is obtained and saved.
"""

import argparse
import os
import sys

import numpy as np
import nibabel as nib
import csv
import datetime


if __name__ == "__main__":

    # -------------------------------------------------------------------------------------------------------- #
    # ----                                       PARSER ARGUMENTS                                         ---- #
    # -------------------------------------------------------------------------------------------------------- #

    # parse command line
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=f'Evaluate the registration of two volumes using the deformation field')

    # path parameters
    p.add_argument('--def-field-path', required=True, help='path to the spinal cord segmentation of the fixed image')
    p.add_argument('--sub-id', required=True, help='id of the subject')

    p.add_argument('--out-file', required=False, default='jacobian_det.csv',
                   help='path to csv summarizing the results obtained')
    p.add_argument('--out-im-path', required=False, default='detJa.nii.gz',
                   help='path to output the volume representative of the determinant of the Jacobian')
    p.add_argument('--append', type=int, required=False, default=1, choices=[0, 1],
                   help="Append results as a new line in the output csv file instead of overwriting it.")

    arg = p.parse_args()

    # -------------------------------------------------------------------------------------------------------- #
    # ----                                LOADING THE DEFORMATION FIELD                                   ---- #
    # -------------------------------------------------------------------------------------------------------- #

    if len(arg.def_field_path.split('.')) > 1:
        def_field = nib.load(arg.def_field_path)
    else:
        def_field = nib.load(f'{arg.def_field_path}.nii.gz')

    ddf = np.array(def_field.get_fdata())

    # -------------------------------------------------------------------------------------------------------- #
    # ----                            COMPUTE THE JACOBIAN AND ITS DETERMINANT                            ---- #
    # -------------------------------------------------------------------------------------------------------- #

    """
    The following code has been adapted from the work of Andjela Dimitrijevic (@Andjelaaaa)
    on https://github.com/polymagic/DLRegistrationFramework
    """

    height, width, depth, time_dim, num_channel = ddf.shape

    num_voxel = (height-4) * (width-4) * (depth-4)

    dx = np.reshape((ddf[:-4, 2:-2, 2:-2, :] - 8*ddf[1:-3, 2:-2, 2:-2, :] + 8*ddf[3:-1, 2:-2, 2:-2, :] - ddf[4:, 2:-2, 2:-2, :])/12.0, [num_voxel, num_channel])
    dy = np.reshape((ddf[2:-2, :-4, 2:-2, :] - 8*ddf[2:-2, 1:-3, 2:-2, :] + 8*ddf[2:-2, 3:-1, 2:-2, :] - ddf[2:-2, 4:, 2:-2, :])/12.0, [num_voxel, num_channel])
    dz = np.reshape((ddf[2:-2, 2:-2, :-4, :] - 8*ddf[2:-2, 2:-2, 1:-3, :] + 8*ddf[2:-2, 2:-2, 3:-1, :] - ddf[2:-2, 2:-2, 4:, :])/12.0, [num_voxel, num_channel])
    J = np.stack([dx, dy, dz], 2)

    det = []
    for voxel in range(num_voxel):
        J[voxel, 0, 0] = J[voxel, 0, 0] + 1
        J[voxel, 1, 1] = J[voxel, 1, 1] + 1
        J[voxel, 2, 2] = J[voxel, 2, 2] + 1
        n_array = J[voxel, :, :]
        det.append(np.linalg.det(n_array))

    comparison = np.where(np.array(det) > 0, 0, np.array(det))  # all negatives values are there and positives become 0
    negative_dets = np.count_nonzero(comparison)
    percentage_negative = 100 * negative_dets / len(det)

    det_Ja = np.reshape(np.array(det), [height - 4, width - 4, depth - 4, 1])
    img = nib.Nifti1Image(det_Ja, def_field.affine)
    nib.save(img, arg.out_im_path)

    res_summary = dict()
    res_summary['subject'] = arg.sub_id
    res_summary['percentage_negative_detJa'] = percentage_negative
    res_summary['median_detJa'] = np.median(det)
    res_summary['mean_detJa'] = np.mean(det)
    res_summary['std_detJa'] = np.std(det)
    res_summary['n_total_detJa'] = len(det)
    res_summary['n_negatives_detJa'] = negative_dets

    # write header (only if append=False)
    if not arg.append or not os.path.isfile(arg.out_file):
        with open(arg.out_file, 'w') as csvfile:
            header = ['Timestamp', 'Subject', 'Percentage_negative_detJa[%]', 'Median_detJa',
                      'Mean_detJa', 'Std_detJa', 'N_total_voxels', 'N_voxels_negatives_detJa']
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()

    # populate data
    with open(arg.out_file, 'a') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        line = list()
        line.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))  # Timestamp
        for val in res_summary.keys():
            line.append(str(res_summary[val]))
        spamwriter.writerow(line)

    sys.exit(0)
