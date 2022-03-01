#!/bin/bash
#
# Register moving volumes to fixed volumes and evaluate the registration by computing the volume overlap of the spinal cord
# segmentations of the volumes involved in the process and the mutual information between these volumes as well
# Specify at the start of the script the contrasts to register together
# Specify if multi-sessions are available for the subjects
#
# ANTs image based deformable registration. This approach is similar to the deep-learning based registration models
# as the registration is image based and includes only deformable registration.
# Mutual information is used as a similarity metric and with a non-linear symmetric normalization (SyN) algorithm
# to compute the transformation. The images are down-sampled by a factor of 2 to allow bigger deformations
# and faster computations and a smooth factor of 0.5 mm is used.
#
#
# Usage:
# sct_run_batch -jobs 1 -path-data bids_dataset -path-out res_registration -script pipe_IMD.sh
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

# FUNCTIONS
# ==============================================================================

# Perform segmentation.
segment(){
  local file="$1"
  local contrast="$2"

  echo "Proceeding with automatic segmentation"
  # Segment spinal cord
  sct_deepseg_sc -i ${file}.nii.gz -c $contrast -qc ${PATH_QC}_seg -qc-subject ${SUBJECT}
}

# SCRIPT STARTS HERE
# ==============================================================================
# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

SUBJECT_ID=$(dirname "$SUBJECT")
SES=$(basename "$SUBJECT")

# PARAMETERS TO SPECIFY
# ==============================================================================
# Choose the name and the extension of the fixed volume (ex: T1w and .nii.gz) and its contrast (ex: t1)
FX_NAME="T1w"
FX_EXT=".nii.gz"
FX_CONTRAST="t1"  # This parameter is used for SC segmentation and should be one of {t1, t2, t2s, dwi}
# Choose the name and the extension of the moving volume (ex: T2w and .nii.gz) and its contrast (ex: t2)
MOV_NAME="T2w"
MOV_EXT=".nii.gz"
MOV_CONTRAST="t2"  # This parameter is used for SC segmentation and should be one of {t1, t2, t2s, dwi}
# Specify if multi-sessions are available per subject: if yes set the value to 1, if no set the value to 0
MULT_SESSIONS=0

# Choose whether to keep all the files created during the process (DEBUGGING=1) (in add_res and seg folders) to debug
# and observe what happened at the different steps of the process or to keep only the two volumes of origin
# and processed/registered (in origin and res folders) (DEBUGGING=0)
DEBUGGING=1
# Choose whether to keep original naming and location of input volumes for the registered volumes. It's recommended to set the value to 1 if the dataset
# will later be used for other tasks, such as segmentation. If the value is 1, the res folder will be removed and the
# registered volumes will be directly present in the anat folder and with the same names as the original volumes
KEEP_ORI_NAMING_LOC=0
# Choose which type of evaluation you want to run to assess the registration results (1 will run the analysis, 0 no)
EVAL_METRICS_ON_SC_SEG=1
EVAL_MI=1
EVAL_JACOBIAN=1
# ==============================================================================

# Go to folder where data will be copied and processed
cd ${PATH_DATA_PROCESSED}
# Copy source images
mkdir -p ${SUBJECT}
rsync -avzh $PATH_DATA/$SUBJECT/ ${SUBJECT}
# Go to anat folder where all structural data are located
echo $PWD
cd ${SUBJECT}/anat/

if [ $MULT_SESSIONS == 1 ]
then
  file_fx_before_proc="${SUBJECT_ID}_${SES}_${FX_NAME}"
  file_mov_before_proc="${SUBJECT_ID}_${SES}_${MOV_NAME}"
else
  file_fx_before_proc="${SES}_${FX_NAME}"
  file_mov_before_proc="${SES}_${MOV_NAME}"
fi

# Do the same preprocessing steps as when using the deep learning-based registration models
# Resample the fixed image to isotropic 1mm resolution
sct_resample -i ${file_fx_before_proc}${FX_EXT} -mm 1x1x1 -o ${file_fx_before_proc}_res1.nii.gz
# Bring the moving image to the same space as the resampled fixed image (for visualization of the results and for the post-registration analysis)
sct_register_multimodal -i ${file_mov_before_proc}${MOV_EXT} -d ${file_fx_before_proc}_res1.nii.gz -o ${file_mov_before_proc}_res1.nii.gz -identity 1

# Perform the image based registration moving --> fixed
# NOTE: We are passing the un-resampled / original moving image as the input to avoid double interpolation
# Only image based deformable registration.
sct_register_multimodal -i ${file_mov_before_proc}${MOV_EXT} -d ${file_fx_before_proc}_res1.nii.gz -o ${file_mov_before_proc}_reg.nii.gz -owarp ${file_mov_before_proc}_warp_field.nii.gz -param step=1,type=im,algo=syn,deformation=1x1x1,metric=MI,smooth=0.5,iter=30,shrink=2

file_fx="${file_fx_before_proc}_res1"
file_mov="${file_mov_before_proc}_res1"
file_mov_reg="${file_mov_before_proc}_reg"
file_warp="${file_mov_before_proc}_warp_field.nii.gz"

if [ $MULT_SESSIONS == 1 ]
then
  sub_id="${SUBJECT_ID}_${SES}"
else
  sub_id="${SES}"
fi

if [ $EVAL_METRICS_ON_SC_SEG == 1 ]
then
  # Segment spinal cord
  segment $file_fx $FX_CONTRAST
  segment $file_mov $MOV_CONTRAST
  segment $file_mov_reg $MOV_CONTRAST

  file_fx_seg="${file_fx}_seg"
  file_mov_seg="${file_mov}_seg"
  file_mov_reg_seg="${file_mov_reg}_seg"
fi

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate smenv
if [ $EVAL_METRICS_ON_SC_SEG == 1 ]
then
  # Compute metrics on SC segmentation overlap before and after registration and save the results in a csv file
  python $PATH_SCRIPT/eval_reg_on_sc_seg.py --fx-seg-path $file_fx_seg --moving-seg-path $file_mov_seg --warped-seg-path $file_mov_reg_seg --sub-id $sub_id --out-file $PATH_DATA_PROCESSED/metrics_on_sc_seg.csv --append 1
fi
if [ $EVAL_MI == 1 ]
then
  # Compute the normalized Mutual Information and save the results in a csv file
  python $PATH_SCRIPT/eval_reg_with_mi.py --fx-im-path $file_fx --moving-im-path $file_mov --warped-im-path $file_mov_reg --sub-id $sub_id --out-file $PATH_DATA_PROCESSED/nmi.csv --append 1
fi
if [ $EVAL_JACOBIAN == 1 ]
then
  # Compute the determinant of the Jacobian and save the results in a csv file
  python $PATH_SCRIPT/eval_reg_with_jacobian.py --def-field-path $file_warp --sub-id ${SES} --out-file $PATH_DATA_PROCESSED/jacobian_det.csv --out-im-path $PATH_DATA_PROCESSED/$SUBJECT/anat/detJa.nii.gz --append 1
fi
conda deactivate

if [ $EVAL_METRICS_ON_SC_SEG == 1 ]
then
  # Generate QC report to assess registration
  sct_qc -i ${file_fx}.nii.gz -s ${file_fx_seg}.nii.gz -d ${file_mov}.nii.gz -p sct_register_multimodal -qc ${PATH_QC}_reg -qc-subject ${SUBJECT}
  sct_qc -i ${file_fx}.nii.gz -s ${file_fx_seg}.nii.gz -d ${file_mov_reg}.nii.gz -p sct_register_multimodal -qc ${PATH_QC}_reg -qc-subject ${SUBJECT}
fi

# Rearrange the different files to obtain an output that has a better structure and is easier to use
# The processed fixed and processed + registered moving volumes are stored in the res folder or directly in anat folder if KEEP_ORI_NAMING_LOC=1
# The original fixed and moving volumes are stored in the origin folder
# If DEBUGGING=1, the additional volumes computed during the process are stored in the add_res folder
# and the segmentations are stored in the seg folder
mkdir origin
mv "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${file_fx_before_proc}${FX_EXT}" "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/origin/${file_fx_before_proc}${FX_EXT}"
mv "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${file_mov_before_proc}${MOV_EXT}" "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/origin/${file_mov_before_proc}${MOV_EXT}"

if [ $MULT_SESSIONS == 1 ]
then
  json_fx_ori="${SUBJECT_ID}_${SES}_${FX_NAME}.json"
  json_mov_ori="${SUBJECT_ID}_${SES}_${MOV_NAME}.json"
else
  json_fx_ori="${SES}_${FX_NAME}.json"
  json_mov_ori="${SES}_${MOV_NAME}.json"
fi

mv "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${json_fx_ori}" "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/origin/${json_fx_ori}"
mv "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${json_mov_ori}" "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/origin/${json_mov_ori}"

mkdir res
mv "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${file_fx}.nii.gz" "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/res/${file_fx}.nii.gz"
mv "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${file_mov_reg}.nii.gz" "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/res/${file_mov_reg}.nii.gz"

if [ $DEBUGGING == 1 ]
then
  if [ $EVAL_METRICS_ON_SC_SEG == 1 ]
  then
    mkdir seg
    filenames_seg=`ls ./*_seg.nii.gz`
    for file in $filenames_seg
    do
       mv "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/$file" "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/seg/$file"
    done
  fi
  mkdir add_res
  filenames=`ls ./*.nii.gz`
  for file in $filenames
  do
     mv "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/$file" "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/add_res/$file"
  done
else
  filenames=`ls ./*.nii.gz`
  for file in $filenames
  do
     rm -f "$file"
  done
fi

if [ $KEEP_ORI_NAMING_LOC == 1 ]
then
  mv "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/res/${file_fx}.nii.gz" "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${file_fx_before_proc}"
  mv "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/res/${file_mov_reg}.nii.gz" "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${file_mov_before_proc}"
  rm -rf "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/res/"
fi

# Verify presence of output files and write log file if error
# ------------------------------------------------------------------------------
if [ $KEEP_ORI_NAMING_LOC == 0 ]
then
  FILES_TO_CHECK=(
    "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/res/${file_fx}.nii.gz"
    "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/res/${file_mov_reg}.nii.gz"
  )
else
  FILES_TO_CHECK=(
    "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${file_fx}.nii.gz"
    "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${file_mov_reg}.nii.gz"
  )
fi

pwd
for file in ${FILES_TO_CHECK[@]}; do
  if [[ ! -e $file ]]; then
    echo "${file} does not exist" >> $PATH_LOG/_error_check_output_files.log
  fi
done

# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"
