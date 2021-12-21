#!/bin/bash
#
# Register T2w volumes to T1w volumes and evaluate the registration by computing the volume overlap of the spinal cord
# segmentations of the volumes involved in the process and the mutual information between these volumes as well
#
# Usage:
# sct_run_batch -jobs 1 -path-data bids_dataset -path-out res_registration -script pipeline_bids_register_evaluate.sh
#
#
# Author: Evan Béal

set -x
# Immediately exit if error
# set -e -o pipefail

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

# Choose the registration model to use (should be in the model folder)
REGISTRATION_MODEL="registration_model.h5"
# Set the following value to 0 to not perform affine registration in any case
MIN_SC_DICE_EXPECTED_PERC=80

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
# Perform processing and registration
python $PATH_SCRIPT/bids_registration.py --model-path $PATH_SCRIPT/model/$REGISTRATION_MODEL --fx-img-path $file_t1_before_proc --mov-img-path $file_t2_before_proc --fx-img-contrast T1w --one-cpu-tf True
conda deactivate

file_t1="${SES}_T1w_proc"
file_t2="${SES}_T2w_proc"
file_t2_reg="${SES}_T2w_proc_reg_to_T1w"

# Segment spinal cord
segment $file_t1 "t1"
segment $file_t2 "t2"
segment $file_t2_reg "t2"

file_t1_seg="${SES}_T1w_proc_seg"
file_t2_seg="${SES}_T2w_proc_seg"
file_t2_reg_seg="${SES}_T2w_proc_reg_to_T1w_seg"

conda activate smenv
# Compute Dice score of SC segmentation overlap before and after registration and save the results in a csv file
python $PATH_SCRIPT/eval_reg_on_sc_seg.py --fx-seg-path $file_t1_seg --moving-seg-path $file_t2_seg --warped-seg-path $file_t2_reg_seg --sub-id ${SES} --out-file $PATH_DATA_PROCESSED/dice_score.csv --append 1 --min-dice $MIN_SC_DICE_EXPECTED_PERC --last-eval 0
RETURN_VAL=$?
conda deactivate

# If the Dice Score obtained after the model's registration is below MIN_SC_DICE_EXPECTED_PERC/100,
# then perform an affine registration before doing the model's registration
if [ $RETURN_VAL == 1 ]
then

  echo Perform affine registration prior to model registration
  # Compute the SC segmentation
  segment $file_t1_before_proc "t1"
  segment $file_t2_before_proc "t2"

  # Set the different filenames for the multimodal affine registration based on the SC segmentation
  file_t1_before_proc="${SES}_T1w.nii.gz"
  file_t2_before_proc="${SES}_T2w.nii.gz"
  file_t1_before_proc_seg="${SES}_T1w_seg.nii.gz"
  file_t2_before_proc_seg="${SES}_T2w_seg.nii.gz"
  t2_affine_reg="${SES}_T2w_aff_reg.nii.gz"

  # Multimodal affine registration based on the SC segmentation
  sct_register_multimodal -i $file_t2_before_proc -iseg $file_t2_before_proc_seg -d $file_t1_before_proc -dseg $file_t1_before_proc_seg -param step=1,type=seg,algo=centermass -o $t2_affine_reg

  # Set the different filenames for the registration using synthmorph approach
  file_t1_before_proc="${SES}_T1w"
  file_t2_before_proc="${SES}_T2w_aff_reg"

  conda activate smenv
  # Perform processing and model's registration
  python $PATH_SCRIPT/bids_registration.py --model-path $PATH_SCRIPT/model/$REGISTRATION_MODEL --fx-img-path $file_t1_before_proc --mov-img-path $file_t2_before_proc --fx-img-contrast T1w --one-cpu-tf False
  conda deactivate

  file_t1="${SES}_T1w_proc"
  file_t2="${SES}_T2w_aff_reg_proc"
  file_t2_reg="${SES}_T2w_aff_reg_proc_reg_to_T1w"

  # Segment spinal cord
  segment $file_t1 "t1"
  segment $file_t2 "t2"
  segment $file_t2_reg "t2"

  file_t1_seg="${SES}_T1w_proc_seg"
  file_t2_seg="${SES}_T2w_aff_reg_proc_seg"
  file_t2_reg_seg="${SES}_T2w_aff_reg_proc_reg_to_T1w_seg"

  conda activate smenv
  # Compute Dice score of SC segmentation overlap before and after registration and save the results in a csv file
  python $PATH_SCRIPT/eval_reg_on_sc_seg.py --fx-seg-path $file_t1_seg --moving-seg-path $file_t2_seg --warped-seg-path $file_t2_reg_seg --sub-id ${SES} --out-file $PATH_DATA_PROCESSED/dice_score.csv --append 1
  # Compute the normalized Mutual Information and save the results in a csv file
  python $PATH_SCRIPT/eval_reg_with_mi.py --fx-im-path $file_t1_seg --moving-im-path $file_t2_seg --warped-im-path $file_t2_reg_seg --sub-id ${SES} --out-file $PATH_DATA_PROCESSED/nmi.csv --append 1
  conda deactivate

else

  conda activate smenv
  # Compute the normalized Mutual Information and save the results in a csv file
  python $PATH_SCRIPT/eval_reg_with_mi.py --fx-im-path $file_t1_seg --moving-im-path $file_t2_seg --warped-im-path $file_t2_reg_seg --sub-id ${SES} --out-file $PATH_DATA_PROCESSED/nmi.csv --append 1
  conda deactivate

fi

# Generate QC report to assess registration
sct_qc -i ${file_t1}.nii.gz -s ${file_t1_seg}.nii.gz -d ${file_t2}.nii.gz -p sct_register_multimodal -qc ${PATH_QC} -qc-subject ${SUBJECT}
sct_qc -i ${file_t1}.nii.gz -s ${file_t1_seg}.nii.gz -d ${file_t2_reg}.nii.gz -p sct_register_multimodal -qc ${PATH_QC} -qc-subject ${SUBJECT}
sct_qc -i ${file_t2}.nii.gz -s ${file_t2_seg}.nii.gz -d ${file_t2_reg}.nii.gz -p sct_register_multimodal -qc ${PATH_QC} -qc-subject ${SUBJECT}

# Verify presence of output files and write log file if error
# ------------------------------------------------------------------------------
FILES_TO_CHECK=(
  "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${SES}_T1w_proc.nii.gz"
  "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${SES}_T2w_proc.nii.gz"
  "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${SES}_T2w_proc_reg_to_T1w.nii.gz"
  "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${SES}_T1w_proc_seg.nii.gz"
  "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${SES}_T2w_proc_seg.nii.gz"
  "${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${SES}_T2w_proc_reg_to_T1w_seg.nii.gz"
)
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
