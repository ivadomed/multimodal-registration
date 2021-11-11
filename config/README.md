## Parameters description
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
- `zero_borders_maps`: boolean to create (True) or not (False) some borders with zero-voxels on label maps (default: true)
- `in_shape`: dimension of label maps produced (default: 128 128 128) (should be divisable by 2 to the power of n, where n is the number of up/down-sampling steps, to ensure that the concatenations are possible)
- `num_labels`: number of different labels in the maps (feature dimension) (default: 26)
- `num_maps`: number of different label maps produced (default: 10)
- `im_scales`: list of relative resolutions at which noise is sampled normally (to produce the label map) (default: 16 32 64)
- `def_scales`: list of relative resolutions at which noise is sampled normally (to produce the deformation field) (default: 16 32 64)
- `im_max_std`: max std for the gaussian distribution of noise in label maps generation (images) (default: 1)
- `def_max_std`: max std for the gaussian dist of noise in label maps generation (def field) (default: 16)
- `add_str`: additional string to specify how the maps are created (default: "26lab_")

### Generation of grayscale images parameters
- `same_subj`: generate image pairs from same label map (default: true)
- `blur_std`: maximum blurring std. dev. (default: 1)
- `gamma`: std. dev. of gamma (default: 0.25)
- `vel_std`: std. dev. of SVF (default: 0.5)
- `vel_res`: SVF scale (default: 16)
- `bias_std`: std. dev. of bias field (default: 0.3)
- `bias_res`: bias scale (default: 40)

### Training parameters
- `gpu`: ID of GPU to use (default: 0)
- `epochs`: training epochs (default: 100) 
- `batch_size`: batch size (default: 1)
- `save_freq`: epochs between model saves (default: 10)
- `bool_init_weights`: boolean to use weights file to initialize with (True) or not (False) (default: false)
- `init_weights`: optional weights file to initialize with (default: "model.h5")
- `reg_param`: regularization weight (default: 1)
- `lr`: learning rate (default: 1e-4)
- `init_epoch`: initial epoch number (default: 0)
- `verbose`: 0 silent, 1 bar, 2 line/epoch (default: 1)

### Network architecture parameters
- `int_steps`: number of integration steps (default: 5)
- `enc`: U-net encoder filters (default: 64 64 64 64)
- `dec`: U-net decoder filters (default: 64 64 64 64 64 64)
