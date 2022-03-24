#!/bin/bash
#
# Apply random affine transformations to all the T2w volumes of a dataset following BIDS convention
#
# Usage:
# sct_run_batch -jobs 1 -path-data bids_dataset -path-out res_registration -script pipe_random_affine_transform_dataset.sh
#
#
# Author: Evan Béal

set -x
# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux, OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Retrieve input params
SUBJECT=$1

# Save script path
PATH_SCRIPT=$PWD

# get starting time:
start=`date +%s`

# SCRIPT STARTS HERE
# ==============================================================================
# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# SUBJECT_ID=$(dirname "$SUBJECT")
SES=$(basename "$SUBJECT")

# Choose whether to keep original naming and location of input volumes for the transformed volumes.
KEEP_ORI_NAMING_LOC=1

# Go to folder where data will be copied and processed
cd ${PATH_DATA_PROCESSED}
# Copy source images
mkdir -p ${SUBJECT}
rsync -avzh $PATH_DATA/$SUBJECT/ ${SUBJECT}
# Go to anat folder where all structural data are located
echo $PWD
cd ${SUBJECT}/anat/

file_mov_before_aff_transfo="${SES}_T2w"

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate smenv
# Transform the volume
python $PATH_SCRIPT/random_affine_transform_dataset.py --mov-img-path $file_mov_before_aff_transfo --sub-id ${SES} --out-file $PATH_DATA_PROCESSED/summary_transform.csv
conda deactivate

file_mov_transformed="${file_mov_before_aff_transfo}_aff_transformed"

if [ $KEEP_ORI_NAMING_LOC == 1 ]
then
  rm -rf "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${file_mov_before_aff_transfo}.nii.gz"
  mv "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${file_mov_transformed}.nii.gz" "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${file_mov_before_aff_transfo}.nii.gz"
fi

# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"
