#!/bin/bash
#
# Register T2w volumes to T1w volumes and evaluate the registration by computing the volume overlap of the spinal cord
# segmentations of the volumes involved in the process and the mutual information between these volumes as well
# The registration is done using two models, with the registered volumes obtained from the first model are used as
# inputs for the second registration model.
#
# Usage:
# sct_run_batch -jobs 1 -path-data bids_dataset -path-out res_registration -script pipeline_bids_register_evaluate_two_steps.sh
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
  folder_contrast="anat"

  echo "Proceeding with automatic segmentation"
  # Segment spinal cord
  sct_deepseg_sc -i ${file}.nii.gz -c $contrast -qc ${PATH_QC} -qc-subject ${SUBJECT}
}

# SCRIPT STARTS HERE
# ==============================================================================
# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

SUBJECT_ID=$(dirname "$SUBJECT")
SES=$(basename "$SUBJECT")

# Choose whether to keep all the files created during the process (DEBUGGING=1) (in add_res and seg folders) to debug
# and observe what happened at the different steps of the process or to keep only the two volumes of origin
# and processed/registered (in origin and res folders) (DEBUGGING=0)
DEBUGGING=1
# Choose whether to keep original naming and location of input volumes for the registered volumes. It's recommended to set the value to 1 if the dataset
# will later be used for other tasks, such as segmentation. If the value is 1, the res folder will be removed and the
# registered volumes will be directly present in the anat folder and with the same names as the original volumes
KEEP_ORI_NAMING_LOC=1

# Choose the registration model to use for the first step (should be in the model folder)
# This model should ideally be more specific to affine registration (the model has learned to deal with regularized deformation fields)
AFFINE_REGISTRATION_MODEL="affine_registration_model.h5"
# Choose the registration model to use for the second step (should be in the model folder)
# This model should ideally be more specific to deformable registration (the model can adjust for small variations anywhere)
# The registered volume resulting from the AFFINE_REGISTRATION_MODEL is used as input for this registration model
DEFORMABLE_REGISTRATION_MODEL="deformable_registration_model.h5"

# Go to folder where data will be copied and processed
cd ${PATH_DATA_PROCESSED}
# Copy source images
mkdir -p ${SUBJECT}
rsync -avzh $PATH_DATA/$SUBJECT/ ${SUBJECT}
# Go to anat folder where all structural data are located
echo $PWD
cd ${SUBJECT}/anat/

file_t1_before_proc="${SES}_T1w"
file_t2_before_proc="${SES}_T2w"

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

conda activate smenv
# ---- First registration step ---- #
# Perform processing and registration
python $PATH_SCRIPT/bids_registration.py --model-path $PATH_SCRIPT/model/$AFFINE_REGISTRATION_MODEL --fx-img-path $file_t1_before_proc --mov-img-path $file_t2_before_proc --fx-img-contrast T1w --one-cpu-tf True

file_t1="${SES}_T1w_proc"
file_t2="${SES}_T2w_proc"
file_t2_reg="${SES}_T2w_proc_reg_to_T1w"

# ---- Second registration step ---- #
# Perform processing and registration
python $PATH_SCRIPT/bids_registration.py --model-path $PATH_SCRIPT/model/$DEFORMABLE_REGISTRATION_MODEL --fx-img-path $file_t1 --mov-img-path $file_t2_reg --fx-img-contrast T1w --one-cpu-tf True --already-preproc 1
conda deactivate

file_t2_reg="${SES}_T2w_proc_reg_to_T1w_proc_reg_to_T1w"

# Segment spinal cord
segment $file_t1 "t1"
segment $file_t2 "t2"
segment $file_t2_reg "t2"

file_t1_seg="${SES}_T1w_proc_seg"
file_t2_seg="${SES}_T2w_proc_seg"
file_t2_reg_seg="${SES}_T2w_proc_reg_to_T1w_proc_reg_to_T1w_seg"

conda activate smenv
# Compute Dice score of SC segmentation overlap before and after registration and save the results in a csv file
python $PATH_SCRIPT/eval_reg_on_sc_seg.py --fx-seg-path $file_t1_seg --moving-seg-path $file_t2_seg --warped-seg-path $file_t2_reg_seg --sub-id ${SES} --out-file $PATH_DATA_PROCESSED/dice_score.csv --append 1
# Compute the normalized Mutual Information and save the results in a csv file
python $PATH_SCRIPT/eval_reg_with_mi.py --fx-im-path $file_t1_seg --moving-im-path $file_t2_seg --warped-im-path $file_t2_reg_seg --sub-id ${SES} --out-file $PATH_DATA_PROCESSED/nmi.csv --append 1
conda deactivate

# Generate QC report to assess registration
sct_qc -i ${file_t1}.nii.gz -s ${file_t1_seg}.nii.gz -d ${file_t2}.nii.gz -p sct_register_multimodal -qc ${PATH_QC} -qc-subject ${SUBJECT}
sct_qc -i ${file_t1}.nii.gz -s ${file_t1_seg}.nii.gz -d ${file_t2_reg}.nii.gz -p sct_register_multimodal -qc ${PATH_QC} -qc-subject ${SUBJECT}
sct_qc -i ${file_t2}.nii.gz -s ${file_t2_seg}.nii.gz -d ${file_t2_reg}.nii.gz -p sct_register_multimodal -qc ${PATH_QC} -qc-subject ${SUBJECT}

# Rearrange the different files to obtain an output that has a better structure and is easier to use
# The processed T1w and processed + registered T2w volumes are stored in the res folder or directly in anat folder if KEEP_ORI_NAMING_LOC=1
# The original T1w and T2w volumes are stored in the origin folder
# If DEBUGGING=1, the additional volumes computed during the process are stored in the add_res folder
# and the segmentations are stored in the seg folder
mkdir origin
mv "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${SES}_T1w.nii.gz" "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/origin/${SES}_T1w.nii.gz"
mv "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${SES}_T2w.nii.gz" "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/origin/${SES}_T2w.nii.gz"
mv "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${SES}_T1w.json" "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/origin/${SES}_T1w.json"
mv "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${SES}_T2w.json" "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/origin/${SES}_T2w.json"

mkdir res
mv "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${SES}_T1w_proc.nii.gz" "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/res/${SES}_T1w_proc.nii.gz"
mv "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${SES}_T2w_proc_reg_to_T1w_proc_reg_to_T1w.nii.gz" "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/res/${SES}_T2w_proc_reg_to_T1w.nii.gz"

if [ $DEBUGGING == 1 ]
then
  mkdir seg
  filenames_seg=`ls ./*_seg.nii.gz`
  for file in $filenames_seg
  do
     mv "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/$file" "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/seg/$file"
  done
  mkdir add_res
  mv "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${SES}_T2w_proc_reg_to_T1w.nii.gz" "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/add_res/${SES}_T2w_proc_reg_to_T1w_after_first_step.nii.gz"
  mv "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${SES}_T2w_proc_field_to_T1w.nii.gz" "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/add_res/${SES}_T2w_proc_field_to_T1w_after_first_step.nii.gz"
  mv "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${SES}_T2w_proc_reg_to_T1w_proc_field_to_T1w.nii.gz" "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/add_res/${SES}_T2w_proc_field_to_T1w.nii.gz"
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
  mv "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/res/${SES}_T1w_proc.nii.gz" "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${SES}_T1w.nii.gz"
  mv "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/res/${SES}_T2w_proc_reg_to_T1w.nii.gz" "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${SES}_T2w.nii.gz"
  rm -rf "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/res/"
fi

# Verify presence of output files and write log file if error
# ------------------------------------------------------------------------------
if [ $KEEP_ORI_NAMING_LOC == 0 ]
then
  FILES_TO_CHECK=(
    "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/res/${SES}_T1w_proc.nii.gz"
    "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/res/${SES}_T2w_proc_reg_to_T1w.nii.gz"
  )
else
  FILES_TO_CHECK=(
    "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${SES}_T1w.nii.gz"
    "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${SES}_T2w.nii.gz"
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
