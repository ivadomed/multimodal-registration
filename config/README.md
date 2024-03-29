## Training parameters description
Description of the different parameters that can be modified/specified in the config file to use `train_synthmorph.py` in order to generate synthetic label maps, to generate synthetic grayscale images and to train a model for multimodal registration.

### Data organization parameters
- `model_dir`: model output directory (default: "models")
- `log_dir`: optional TensorBoard log directory (default: "logs")
- `bool_sub_dir`: boolean to specify the use of an optional subfolder for logs and model saves (default: false)
- `sub_dir`: optional subfolder for logs and model saves (default: "train_ex")

### Generation of label maps parameters
- `gen_label_only`: boolean to specify that only the label maps generation should be done (True) or not (False) and not the rest of the training process (default: false)
- `gen_label`: boolean to generate (True) or not (False) label maps from noise distribution (default: true)
- `save_label`: boolean to save the generated label locally (True) or not (False) (default: true)
- `label_dir`: directory pointing to input label maps if `gen_label` is False, if `gen_label` is True and `save_label` is True, directory used to save the labels (default: labels)
- `zero_borders_maps`: boolean to create (True) or not (False) some borders with zero-voxels on label maps (default: false)
- `zero_borders_maps_val`: boolean to create (True) or not (False) some borders with zero-voxels on label maps of the validation dataset (default: false)
- `zero_bord_scale`: integer to determine the maximum width of the added zero-borders (1 / zero_bord_scale) (default: 8)
- `zero_bord_frac`: fraction of the time [0, 1] that the images will enter in the zero-padding process (used only if `zero_borders_map` or `zero_borders_map_val` is True) (default: 0.5)
- `in_shape`: dimension of label maps produced (default: 160 160 192) (should be divisable by 2 to the power of n, where n is the number of up/down-sampling steps, to ensure that the concatenations are possible)
- `num_labels`: number of different labels in the maps (feature dimension) (default: 26)
- `num_maps`: number of different label maps produced (default: 100)
- `im_scales`: list of relative resolutions at which noise is sampled normally (to produce the label map) (default: 16 32 64)
- `def_scales`: list of relative resolutions at which noise is sampled normally (to produce the deformation field) (default: 8 16 32)
- `im_max_std`: max std for the gaussian distribution of noise in label maps generation (images) (default: 1)
- `def_max_std`: max std for the gaussian dist of noise in label maps generation (def field) (default: 3)
- `add_str`: additional string to specify how the maps are created (default: "26lab_")

### Generation of grayscale images parameters
- `same_subj`: generate image pairs from same label map (default: true)
- `blur_std`: maximum blurring std. dev. (default: 1)
- `gamma`: std. dev. of gamma (default: 0.25)
- `vel_std`: std. dev. of SVF (default: 3)
- `vel_res`: SVF scale (default: 16)
- `bias_std`: std. dev. of bias field (default: 0.3)
- `bias_res`: bias scale (default: 40)

### Training parameters
- `gpu`: ID of GPU to use (default: 0)
- `epochs`: training epochs (default: 600) 
- `batch_size`: batch size (default: 1)
- `train_frac`: fraction of the label maps that will be included in the training dataset, the other part will form the validation dataset (default: 0.8)
- `batch_size_val`: batch size of the validation dataset (default: 1)
- `save_freq`: epochs between model saves (default: 100)
- `bool_init_weights`: boolean to use weights file to initialize with (True) or not (False) (default: false)
- `init_weights`: optional weights file to initialize with (default: "model.h5")
- `reg_param`: regularization weight (default: 1)
- `lr`: learning rate (default: 1e-4)
- `init_epoch`: initial epoch number (default: 0)
- `verbose`: 0 silent, 1 bar, 2 line/epoch (default: 1)

### Network architecture parameters
- `int_steps`: number of integration steps (default: 5)
- `int_res`: resolution (relative voxel size) of the flow field during vector integration (default: 2)
- `svf_res`: resolution (relative voxel size) of the predicted SVF (default: 2)
- `enc`: U-net encoder filters (default: 64 64 64 64)
- `dec`: U-net decoder filters (default: 64 64 64 64 64 64)

---
## Inference parameters description
Description of the different parameters that can be modified/specified in the config_inference file to use `3d_reg.py` and the different shell scripts like `bids_register_evaluate.sh`. Some of these parameters need to be the same as the ones used to train the registration model, whereas the first parameters others are specific to the strategy wanted for the registratrion during inference time. You can choose between doing the inference directly on the whole volume (better accuracy but greater computational resources needed) or on subvolumes of the size specified.

### Parameters independent from the trained registration model
- `use_subvol`: boolean to decide whether to use the whole volume as input of the registration model (False) (better results, more computational resources needed) or to create subvolumes to use as input of the registration model (True) before constructing the warping field that will be applied to the whole volume (default: false)
- `subvol_size`: the size of the subvolumes used (if `use_subvol` is true). Need to be a list of 3 elements representing the size used for each dimension (default: [80, 80, 96])
- `min_perc_overlap`: the minimum percentage of overlap of the subvolumes (if `use_subvol` is true). Can be in percent (ex: 10) or in fraction (ex: 0.1) (default: 0.1)
- `warp_interpolation`: the interpolation to use to get the registered volume from the warping field outputted by the registration model. Can be "linear" or "nearest" (for the nearest neighbor interpolation) (default: "linear")
- `resample_interpolation`: interpolation method used to resample the volumes to a 1 mm isotropic resolution during preprocessing. Can be "linear", “spline” or "nearest" (for the nearest neighbor interpolation) (default: "linear")


### Parameters that need to be similar to the ones used to train the registration model
- `int_steps`: number of integration steps (default: 5)
- `int_res`: resolution (relative voxel size) of the flow field during vector integration (default: 2)
- `svf_res`: resolution (relative voxel size) of the predicted SVF (default: 2)
- `enc`: U-net encoder filters (default: [256, 256, 256, 256])
- `dec`: U-net decoder filters (default: [256, 256, 256, 256, 256, 256])
