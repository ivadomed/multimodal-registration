# Multimodal Registration

Contrast-agnostic registration based on [SynthMorph](https://arxiv.org/pdf/2004.10282.pdf). If you use part of this code, or if you use the model, please cite:

> Beal E, Cohen-Adad J. Contrast-agnostic deep learning–based registration pipeline: Validation in spinal cord multimodal MRI data. Aperture Neuro. Published online July 3, 2023. doi:10.52294/f662441d-2678-4683-8a8c-6ad7be2c4b29

Registration pipelines (including preprocessing, registration, post processing, evaluation) have been developed for the registration of multimodal 3D MR images, focusing on the spinal cord. They provide easy-to-use, accurate and fast solution for multimodal 3D registration. A sketch of the [cascaded pipeline](#registration--evaluation-pipeline-for-large-displacements-two-steps-approach) is represented in the following figure.

<img width="1000" alt="pipe_description" src="https://user-images.githubusercontent.com/32447627/160001735-8b87e1bd-0ae8-4c30-b12c-5ea7f0ea3938.png">

## Table of contents

[Description](#description)  
[Dependencies](#dependencies)  
[Getting started](#getting-started)  
[Training an agnostic registration model](#training-an-agnostic-registration-model)  
[Volumes registration](#volumes-registration)  
[Generate and apply a deformation field](#generate-and-apply-a-deformation-field)  
[Registration & Evaluation pipeline](#registration--evaluation-pipeline)  
[Registration & Evaluation pipeline with optional affine registration step](#registration--evaluation-pipeline-with-optional-affine-registration-step)  
[Registration & Evaluation pipeline for large displacements (Two steps approach)](#registration--evaluation-pipeline-for-large-displacements-two-steps-approach)


## Description

This repository contains a file `train_synthmorph.py` allowing to easily use the SynthMorph method to train a contrast-invariant registration model from a config file (.json). Some additional files are provided to perform registration starting from volumes of any size (pre-processing steps included in the registration file) and to be able to generate unregistered volumes by applying a deformation field synthesized from noise distribution.

In addition, different pipelines (.sh shell script) are provided: `pipeline_bids_register_evaluate.sh`, `pipeline_bids_register_evaluate_opt_affine.sh`, ...
These pipelines offer a framework for the registration of pair of images of any modality (contrasts) for each subject of any dataset following the Brain Imaging Data Structure ([BIDS](https://bids.neuroimaging.io/)) convention (can be two different modalities). Evaluation tools are included in the pipeline (using different features from the Spinal Cord Toolbox ([SCT](https://spinalcordtoolbox.com/) notably) to assess the registration results, focusing on the spinal cord.

This strategy of learning contrast-invariant registration is explored in regard of the IvadoMed’s issue [#659](https://github.com/ivadomed/ivadomed/issues/659) on multimodal segmentation tasks with non-coregistered data. Adding a contrast agnostic registration model as a preprocessing step in the IvadoMed’s pipeline may enable the use of multimodal data for segmentation tasks even when the data are not yet registered.    

## Dependencies

- [Voxelmorph](https://github.com/voxelmorph/voxelmorph) commit: 52dd120f3ae9b0ab0fde5d0efe50627a4528bc9f
- [Neurite](https://github.com/adalca/neurite) commit: c7bb05d5dae47d2a79e0fe5a8284f30b2304d335
- [Pystrum](https://github.com/adalca/pystrum) commit: 8cd5c483195971c0c51e9809f33aa04777aa35c8
- [SCT](https://spinalcordtoolbox.com/) version: [5.4](https://github.com/spinalcordtoolbox/spinalcordtoolbox/releases/tag/5.4)

## Getting started

This repo has been tested with Python 3.9, Tensorflow 2.7.0 and Keras 2.7.0. Follow the steps below to use this repo:
1. Clone the project repository: `git clone https://github.com/ivadomed/multimodal-registration.git`
2. In the project folder, clone the repositories that are used by this project:
```
cd multimodal-registration/
git clone https://github.com/adalca/pystrum
git clone https://github.com/adalca/neurite
git clone https://github.com/voxelmorph/voxelmorph.git
```
3. Create a conda environment, activate it and install the required packages:
```
conda create -y --name smenv python=3.9
conda activate smenv
cd pystrum
pip install .
cd ../neurite
pip install .
cd ../voxelmorph
pip install .
cd ..
conda install nilearn
conda install -c conda-forge tensorflow
```
4. Upgrade tensorflow: `pip install --upgrade tensorflow`
5. Check the sections below for how to use the different features available

## Training an agnostic registration model

The file `train_synthmorph.py` allows you to train a contrast-invariant registration model. All the different steps described in the [SynthMorph paper](https://arxiv.org/pdf/2004.10282.pdf) are performed from this file. 
The parameters used for the generation of label maps, grayscale images and to train the registration model should be specified in a config file. An example is provided with the `config/config.json` file. A description of the different parameters can be found in `config/README.md`.

The main differences between the file in this repo and the `train_synthmorph.py` file that is available in VoxelMorph repo are the possibility to generate label maps directly from this file and an additional cropping/zero-padding step in the generation of label maps to render the model robust to this situation. 

To train a model based on the config file provided:
```
python train_synthmorph.py --config-path config/config.json
```

## Volumes registration

The file `3d_reg.py` allows you to load a trained registration model and register two images together. It includes a preprocessing step to scale the volumes and set them to an isotropic resolution of 1 mm so they can be used by the model. Some parameters of the registration model used need to be specified in a config file, where you can also choose to do the inference directly on the whole volume (better accuracy but greater computational resources needed) or on subvolumes of the size specified with the parameter `subvol_size`.

To perform volumes registration:
```
python 3d_reg.py --model-path model/model.h5 --config-path config/config_inference.json --fx-img-path data/t1.nii.gz --mov-img-path data/t2.nii.gz
```

You can dowload pretrained SynthMorph registration models provided on the [VoxelMorph repository](https://github.com/voxelmorph/voxelmorph) by clicking on the links below:
- ["shapes" variant](https://surfer.nmr.mgh.harvard.edu/ftp/data/voxelmorph/synthmorph/shapes-dice-vel-3-res-8-16-32-256f.h5)
- ["brains" variant](https://surfer.nmr.mgh.harvard.edu/ftp/data/voxelmorph/synthmorph/brains-dice-vel-0.5-res-16-256f.h5)

## Generate and apply a deformation field

The file `gen_apply_def_field.py` takes as input a volume and some parameters to generate a deformation field. It then generates a deformation field based on the specified parameters following the Perlin noise strategy used in synthmorph. This deformation field is then applied to the input volume to obtain a moved object. The moved volume as well as the deformation field generated are saved to the paths specified.

To generate a deformation field and deform a volume with it:
```
python gen_apply_def_field.py --im-path data/t2.nii.gz
```

## Registration & Evaluation pipeline

A pipeline for the registration of a moving volume of a specified modality/contrast to a fixed volume of another specified modality/contrast for each subject of any dataset following Brain Imaging Data Structure ([BIDS](https://bids.neuroimaging.io/)) convention is provided with the shell script `pipeline_bids_register_evaluate.sh`. 
For each subject of the dataset, the moving volume will be registered to the fixed volume using a registration model which name should be defined in the shell script and that should be located in the `model/` folder and a config file for the inference parameters that needs to be in the `config/` folder. The name/contrast/extension of the moving and fixed volumes should also be specified. Everything that needs to be defined is in the section `PARAMETERS TO SPECIFY` of the shell script. Once this is done, the whole process will run automatically.

The first part of the pipeline concerns the preprocessing and registration steps and leads to the creation of 3 new files for each subject: `sub-xx_[contrast_fixed]_proc.nii.gz`, `sub-xx_[contrast_moving]_proc.nii.gz` and `sub-xx_[contrast_moving]_proc_reg_to_[contrast_fixed].nii.gz`. It is done with the file `bids_registration.py`.  

In the second part of the pipeline, these 3 files are used to compute some measurements and obtain a QC report in order to have a an idea of the registration performance. One measurement, the normalized Mutual Information is computed directly on the files obtained with the registration process (first part). It is done with the file `eval_reg_with_mi.py` and results in the file `nmi.csv` that summarises the results obtained for the different comparisons done. 

The second measurement is representative of the spinal cord overlap. To compute this value, the segmentation of the spinal cord should be obtained. This is done with the [`sct_deepseg_sc` feature](https://spinalcordtoolbox.com/user_section/command-line.html#sct-deepseg-sc) of the Spinal Cord Toolbox ([SCT](https://spinalcordtoolbox.com/)) software.  
The spinal cord segmentations are saved and used to compute the volume overlap (Dice score), and other metrics like the Jaccard index or sensitivity, with the file `eval_reg_on_sc_seg.py`. The results are saved in the file `metrics_on_sc_seg.csv` that summarises the results obtained for the different comparisons done.  

Additionally, a Quality Control (QC) report is generated using [`sct_qc`](https://spinalcordtoolbox.com/user_section/command-line.html#sct-qc) from SCT allowing to control the spinal cord segmentations as well as the spinal cord registration. This report takes the form of a `.html` file and can be found at `qc/index.html` in your result folder.

Eventually, the Jacobian determinants of the warping field's displacement vectors are computed to get a visualization of the transformations applied to the voxels of the moving image and determine whether each individual voxel expands, compresses or folds. A `jacobian_det.csv` file is also outputted to summarize the percentage of folding voxels notably. These computations are done with the file `eval_reg_with_jacobian.py`.

<img width="1000" alt="Capture d’écran 2022-02-16 à 14 19 12" src="https://user-images.githubusercontent.com/32447627/154340638-0ae286d4-8c8e-4838-9b56-22035bb049e3.png">

The files obtained during the process (segmentation, processed volumes or deformation fields) are organised into different folders. Two parameters at the beginning of the shell script are monitoring the organisation of the output files in the `anat` folder:
- `DEBUGGING` 
    - if set to 1, all the files are saved and stored into 4 different folders: `res`, `origin`, `add_res` and `seg`.
    - if set to 0, only the input (original) volumes and the registered ones are saved in the folders `origin` and `res` respectively.
- `KEEP_ORI_NAMING_LOC`
    - if set to 1, the registered volumes are saved with the same name and path as the input volumes. Therefore, the `res` folder is deleted. This may be useful if we want to use the dataset for additional computations once the volumes have been registered.
    - if set to 0, nothing happen. 

**To run the shell script**, [`sct_run_batch`](https://spinalcordtoolbox.com/user_section/command-line.html#sct-run-batch) from SCT is used.  
In the project directory, if your BIDS dataset is in the same directory in the `bids_dataset` folder, you can execute the following command to run the registration and evaluation pipeline:
```
sct_run_batch -jobs 1 -path-data bids_dataset -path-out res_registration -script pipeline_bids_register_evaluate.sh
```

⚠️ To use this pipeline you should [install SCT](https://spinalcordtoolbox.com/user_section/installation.html) and have it active in your working environment when running the shell script. The script has been tested with SCT version [5.4](https://github.com/spinalcordtoolbox/spinalcordtoolbox/releases/tag/5.4). 

## Registration & Evaluation pipeline with optional affine registration step

After having followed the same process as the one described in the above section, if the registration result obtained for a certain subject is not as high as desired (based on the Dice score of the spinal cord segmentations), an affine (rigid) registration is done on the input volumes before performing once again the deformable registration with the registration model specified in the shell script. The new results will be assessed with the second part of the pipeline and only these new results will be saved in the different files.
Whether to enter this path of the pipeline or not is monitored by the Dice score obtained on the spinal cord segmentations and the limit fixed in the bash script with the parameter `MIN_SC_DICE_EXPECTED_PERC`. This latter can be set to 0 to avoid using the affine registration on any subjects.
The addition of affine registration in the pipeline is thus optional and can be used as desired by the user to optimize the trade-off between additional computation time and improved registration results for subjects with volumes that are initially in different affine spaces.

The affine registration step is done with [`sct_register_multimodal`](https://spinalcordtoolbox.com/user_section/command-line.html#sct-register-multimodal) from the [Spinal Cord Toolbox](https://spinalcordtoolbox.com/index.html) and using the centermass algorithm, which is a slice wise center of mass alignment done on the spinal cord segmentations (computed on the original input volumes).

<img width="1000" alt="Capture d’écran 2022-02-16 à 14 19 17" src="https://user-images.githubusercontent.com/32447627/154340689-04414229-5b5f-465d-8ff8-fc6f9dba3a6e.png">

**To run the shell script**, [`sct_run_batch`](https://spinalcordtoolbox.com/user_section/command-line.html#sct-run-batch) from SCT is used.  
In the project directory, if your BIDS dataset is in the same directory in the `bids_dataset` folder, you can execute the following command to run the registration and evaluation pipeline:
```
sct_run_batch -jobs 1 -path-data bids_dataset -path-out res_registration -script pipeline_bids_register_evaluate_opt_affine.sh
```

⚠️ To use this pipeline you should [install SCT](https://spinalcordtoolbox.com/user_section/installation.html) and have it active in your working environment when running the shell script. The script has been tested with SCT version [5.4](https://github.com/spinalcordtoolbox/spinalcordtoolbox/releases/tag/5.4). 

## Registration & Evaluation pipeline for large displacements (Two steps approach)

This pipeline (`pipeline_bids_register_evaluate_two_steps.sh`) is an updated version of the original pipeline (`pipeline_bids_register_evaluate.sh`) that uses two registration models successively to register a moving volume to fixed volume for each subject of any dataset following BIDS convention. This approach of using two successive registration models (the registered volume obtained from the first registration model is used as an input for the second model) improves the registration accuracy, especially for largely displaced volumes.  
The first model aims to ensure that the two volumes to register are well aligned (similar to a rigid or affine registration) whereas the second model refines the registration. The two models are outputting a deformation field and are therefore doing deformable registration. However, the two models used should be trained with different characteristics. The first one should learn to register data deformed with a smooth field whereas the second model should learn on data with a lot of small deformations everywhere. This can be done for example by setting the `vel_res` parameter of the config file (when training the registration model) to `[32, 64]` for the first model and to `16` for the second model. 
This approach worked well to register data that have been randomly affine transformed (translation, scaling and rotation) which is not necessarily the case when using a single registration model.  
To use this approach, you need two registration models and specify them in the `pipeline_bids_register_evaluate_two_steps.sh` file. 

<img width="1000" alt="Capture d’écran 2022-02-16 à 14 19 23" src="https://user-images.githubusercontent.com/32447627/154340725-dddb4098-3c63-49c1-b6e9-85f7e01d67e7.png">

**To run the shell script**, [`sct_run_batch`](https://spinalcordtoolbox.com/user_section/command-line.html#sct-run-batch) from SCT is used.  
In the project directory, if your BIDS dataset is in the same directory in the `bids_dataset` folder, you can execute the following command to run the registration and evaluation pipeline:
```
sct_run_batch -jobs 1 -path-data bids_dataset -path-out res_registration -script pipeline_bids_register_evaluate_two_steps.sh
```

⚠️ To use this pipeline you should [install SCT](https://spinalcordtoolbox.com/user_section/installation.html) and have it active in your working environment when running the shell script. The script has been tested with SCT version [5.4](https://github.com/spinalcordtoolbox/spinalcordtoolbox/releases/tag/5.4). 
