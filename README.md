# Multimodal Registration

Repository for training and using a contrast agnostic registration model based on the work done in [SynthMorph](https://arxiv.org/pdf/2004.10282.pdf). The contrast agnostic registration model may later be used in [IvadoMed’s pipeline](https://ivadomed.org/). 

## Description

This repository contains a file `train_synthmorph.py` allowing to easily use the SynthMorph method to train a contrast-invariant registration model from a config file. The code has also been slightly modified to try to adapt the model to zero-padded volumes. Some additional files are provided to perform registration starting from volumes of any size (pre-processing step included in the registration file) and to be able to generate unregistered volumes by applying a deformation field synthesized from noise distribution.

This strategy of learning contrast-invariant registration is explored in regard of the IvadoMed’s issue [#659](https://github.com/ivadomed/ivadomed/issues/659) on multimodal segmentation tasks with non-coregistered data. Adding a contrast agnostic registration model as a preprocessing step in the IvadoMed’s pipeline may enable the use of multimodal data for segmentation tasks even when the data are not yet registered.    

## Getting started

This repo has been tested with Python 3.9. Follow the steps below to use this repo:
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

The file `3d_reg.py` allows you to load a trained registration model and register two images together. It includes a preprocessing step to transform the volumes to the dimensions required by the model.

To perform volumes registration:
```
python 3d_reg.py --model-path model/model.h5 --fx-img-path data/t1 --mov-img-path data/t2
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

A pipeline for T2w volume registration to T1w volume for each subject of any dataset following Brain Imaging Data Structure ([BIDS](https://bids.neuroimaging.io/)) convention is provided with the shell script `pipeline_bids_register_evaluate.sh`. 
For each subject of the dataset, the T2w volume will be registered to the T1w volume using a registration model which name should be specified in the shell script and that should be located in the `model/` folder. This first part of the pipeline leads to the creation of 3 new files for each subject: `sub-xx_T1w_proc.nii.gz`, `sub-xx_T2w_proc.nii.gz` and `sub-xx_T2w_proc_reg_to_T1w.nii.gz`. It is done with the file `bids_registration.py`.  

In the second part of the pipeline, these 3 files are used to compute some measurements and obtain a QC report in order to have a an idea of the registration performance. One measurement, the normalized Mutual Information is computed directly on the files obtained with the registration process (first part). It is done with the file `eval_reg_with_mi.py` and results in the file `nmi.csv` that summarises the results obtained for the different comparisons done. 

The second measurement is representative of the spinal cord overlap. To compute this value, the segmentation of the spinal cord should be obtained. This is done with the [`sct_deepseg_sc` feature](https://spinalcordtoolbox.com/user_section/command-line.html#sct-deepseg-sc) of the Spinal Cord Toolbox ([SCT](https://spinalcordtoolbox.com/)) software.  
The spinal cord segmentations are saved and used to compute the volume overlap (Dice score) with the file `eval_reg_on_sc_seg.py`. The results are saved in the file `dice_score.csv` that summarises the results obtained for the different comparisons done.  

Additionally, a Quality Control (QC) report is generated using [`sct_qc`](https://spinalcordtoolbox.com/user_section/command-line.html#sct-qc) from SCT allowing to control the spinal cord segmentations as well as the spinal cord registration. This report takes the form of a `.html` file and can be found at `qc/index.html` in your result folder.

<img width="900" alt="Capture d’écran 2021-12-03 à 17 30 01" src="https://user-images.githubusercontent.com/32447627/144681407-635ad819-be82-41de-acee-b573ab31aba5.png">

To run the shell script, [`sct_run_batch`](https://spinalcordtoolbox.com/user_section/command-line.html#sct-run-batch) from SCT is used.  
In the project directory, if your BIDS dataset is in the same directory in the `bids_dataset` folder, you can execute the following command to run the registration and evaluation pipeline:
```
sct_run_batch -jobs 1 -path-data bids_dataset -path-out res_registration -script pipeline_bids_register_evaluate.sh
```

⚠️ To use this pipeline you should [install SCT](https://spinalcordtoolbox.com/user_section/installation.html) and have it active in your working environment when running the shell script. The script has been tested with SCT version [5.4](https://github.com/spinalcordtoolbox/spinalcordtoolbox/releases/tag/5.4). 


