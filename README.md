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
cd synthmorph/
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
conda install -c conda-forge tensorflow
pip install --upgrade tensorflow
```
4. Check the sections below for how to use the different features available

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
python 3d_reg.py --model-path model/model.h5 --fx-img-path data/t1 --mov-img-path data/t2 --out-img-path res/t2_warped_t1 --def-field-path res/t2_def_field_t1
```

## Generate and apply a deformation field

The file `gen_apply_def_field.py` takes as input a volume and some parameters to generate a deformation field. It then generates a deformation field based on the specified parameters following the Perlin noise strategy used in synthmorph. This deformation field is then applied to the input volume to obtain a moved object. The moved volume as well as the deformation field generated are saved to the paths specified.

To generate a deformation field and deform a volume with it:
```
python gen_apply_def_field.py --im-path data/t2.nii.gz --out-im-path moved_t2.nii.gz --out-def-path def_field_moved_t2.nii.gz --def-scales 16 32 64 --def-max-std 3 --interp linear
```
