"""
File to train a synthmorph model.
The differences with the file provided in the voxelmorph repository include:
    - the possibility to generate label maps directly from this file
    - the possibility to specify the input image size on which the model will be trained
    - an additional cropping/zero-padding step in the generation of label maps to render
      the model robust to this situation
    - the parameters specific to the generation of label maps, grayscale images and of
      the training of the model are specified in a config file

"""


import sys
import os
import argparse
import contextlib
import tqdm
import json

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import nibabel as nib

import neurite as ne
import voxelmorph as vxm


def generate_label_maps(in_shape, num_labels, num_maps, im_scales, def_scales, im_max_std, def_max_std,
                        save_label, label_dir, add_str=''):
    """
    Function to generate label maps from noise distribution

    Args:
        in_shape: dimension of label maps produced (ex: [256, 256, 256])
        num_labels: number of different labels in the maps (feature dimension)
        num_maps: number of different label maps produced
        im_scales: list of relative resolutions at which noise is sampled normally to produce noise images
        def_scales: list of relative resolutions at which noise is sampled normally to deform the noise images
        im_max_std: max std for the gaussian distribution of noise in label maps generation (images)
        def_max_std: max std for the gaussian distribution of noise in label maps generation (deformation field)
        save_label: boolean to save the generated label locally (True) or not (False)
        label_dir: path where the label maps will be saved

    Returns:
        label_maps: list of all the label maps produced

    """
    label_maps = []

    num_dim = len(in_shape)

    for _ in tqdm.tqdm(range(num_maps)):
        # Draw image and warp.
        im = ne.utils.augment.draw_perlin(
            out_shape=(*in_shape, num_labels),
            scales=im_scales, max_std=im_max_std,
        )
        warp = ne.utils.augment.draw_perlin(
            out_shape=(*in_shape, num_labels, num_dim),
            scales=def_scales, max_std=def_max_std,
        )

        # Transform and create label map.
        im = vxm.utils.transform(im, warp)
        lab = tf.argmax(im, axis=-1)
        label_maps.append(np.uint8(lab))

    if save_label:
        os.makedirs(label_dir, exist_ok=True)
        # save the label maps in the nii.gz format if 3d or in .png if 2d
        if num_dim == 3:
            for i, lab_map in enumerate(label_maps):
                ni_img = nib.Nifti1Image(lab_map, affine=np.eye(4))
                nib.save(ni_img, os.path.join(label_dir, f'label_map_{add_str}{i + 1}.nii.gz'))
        else:
            for i, lab_map in enumerate(label_maps):
                plt.imsave(os.path.join(label_dir, f'label_map_{add_str}{i + 1}.png'), lab_map)

    return label_maps


def set_random_zero_borders(im, scale=8):
    """
    Function to mimic cropped images/volumes that are zero-padded to a certain dimension by
    setting pixels/voxels of borders with a random width to zero

    Parameters:
        im: input image/volume on which zero-borders would be added
        scale: (int) Determine the maximum width of the added zero-borders (1/scale)
    """
    dim_im = len(im.shape) - 1
    out_im = np.zeros_like(im)

    x_lim, y_lim = im.shape[0], im.shape[1]

    x_min = np.random.choice([0, np.random.randint(0, x_lim // scale)])
    x_max = np.random.choice([np.random.randint((scale - 1) * x_lim // scale, x_lim), x_lim])

    y_min = np.random.choice([0, np.random.randint(0, y_lim // scale)])
    y_max = np.random.choice([np.random.randint((scale - 1) * y_lim // scale, y_lim), y_lim])

    if dim_im == 3:
        z_lim = im.shape[2]
        z_min = np.random.choice([0, np.random.randint(0, z_lim // scale)])
        z_max = np.random.choice([np.random.randint((scale - 1) * z_lim // scale, z_lim), z_lim])

        out_im[x_min:x_max, y_min:y_max, z_min:z_max, 0] = im[x_min:x_max, y_min:y_max, z_min:z_max, 0]
    else:
        out_im[x_min:x_max, y_min:y_max, 0] = im[x_min:x_max, y_min:y_max, 0]

    return out_im


def gen_synthmorph_eb(label_maps, batch_size=1, same_subj=False, flip=True,
                      random_zero_borders=True, scale_zero_borders=8):
    """
    Generator for SynthMorph registration.

    Parameters:
        labels_maps: List of pre-loaded ND label maps, each as a NumPy array.
        batch_size: Batch size. Default is 1.
        same_subj: Whether the same label map is returned as the source and target for further
            augmentation. Default is False.
        flip: Whether axes are flipped randomly. Default is True.
        random_zero_borders: Whether to create zero-borders on label maps to mimic zero-padding
        scale_zero_borders: (int) Determine the maximum width of the added zero-borders (1/scale_zero_borders)
    """
    in_shape = label_maps[0].shape
    num_dim = len(in_shape)

    # "True" moved image and warp, that will be ignored by SynthMorph losses.
    void = np.zeros((batch_size, *in_shape, num_dim), dtype='float32')

    rand = np.random.default_rng()
    prop = dict(replace=False, shuffle=False)
    while True:
        ind = rand.integers(len(label_maps), size=2 * batch_size)
        x = [label_maps[i] for i in ind]

        if same_subj:
            x = x[:batch_size] * 2
        x = np.stack(x)[..., None]

        if flip:
            axes = rand.choice(num_dim, size=rand.integers(num_dim + 1), **prop)
            x = np.flip(x, axis=axes + 1)

        src = x[:batch_size, ...]
        trg = x[batch_size:, ...]

        if random_zero_borders:
            for i, trg_im in enumerate(trg):
                # TODO maybe modify the probability (50% for the moment) or create a param
                if np.random.choice([True, False]):
                    trg[i, ...] = set_random_zero_borders(trg_im, scale_zero_borders)
                if np.random.choice([True, False]):
                    src[i, ...] = set_random_zero_borders(src[i, ...], scale_zero_borders)

        yield [src, trg], [void] * 2


if __name__ == "__main__":

    # -------------------------------------------------------------------------------------------------------- #
    # ----                               SPECIFY AND LOAD THE CONFIG FILE                                 ---- #
    # -------------------------------------------------------------------------------------------------------- #

    # parse command line
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=f'Train a SynthMorph model on images synthesized from label maps.')

    # data organization parameters
    p.add_argument('--config-path', default='config/config.json',
                   help='config file with the training parameters specified')

    arg = p.parse_args()

    with open(arg.config_path) as config_file:
        data = json.load(config_file)

    # -------------------------------------------------------------------------------------------------------- #
    # ----                                 PREPARING THE ENVIRONMENT                                      ---- #
    # -------------------------------------------------------------------------------------------------------- #

    # TensorFlow handling
    device, nb_devices = vxm.tf.utils.setup_device(data['gpu'])
    assert np.mod(data['batch_size'], nb_devices) == 0, \
        f'batch size {data["batch_size"]} not a multiple of the number of GPUs {nb_devices}'
    assert tf.__version__.startswith('2'), f'TensorFlow version {tf.__version__} is not 2 or later'

    # -------------------------------------------------------------------------------------------------------- #
    # ----                     LOADING/GENERATING LABEL MAPS FROM NOISE DISTRIBUTION                      ---- #
    # -------------------------------------------------------------------------------------------------------- #

    if data['gen_label']:
        label_maps = generate_label_maps(data['in_shape'], data['num_labels'], data['num_maps'], data['im_scales'],
                                         data['def_scales'], data['im_max_std'], data['def_max_std'],
                                         data['save_label'], data['label_dir'], data['add_str'])
        labels_in = np.unique(label_maps)
    else:
        labels_in, label_maps = vxm.py.utils.load_labels(data['label_dir'])

    np.random.seed(42)
    np.random.shuffle(label_maps)
    label_maps_tr, label_maps_val = np.split(label_maps, [int(len(label_maps)*data['train_frac'])])

    gen_tr = gen_synthmorph_eb(
        label_maps_tr,
        batch_size=data['batch_size'],
        same_subj=data['same_subj'],
        flip=True,
        random_zero_borders=data['zero_borders_maps'],
        scale_zero_borders=data['zero_bord_scale']
    )

    gen_val = gen_synthmorph_eb(
        label_maps_val,
        batch_size=data['batch_size_val'],
        same_subj=data['same_subj'],
        flip=True,
        random_zero_borders=data['zero_borders_maps_val'],
        scale_zero_borders=data['zero_bord_scale']
    )

    in_shape = label_maps[0].shape
    labels_out = labels_in

    if data['gen_label_only']:
        sys.exit(0)

    # -------------------------------------------------------------------------------------------------------- #
    # ----                                    PREPARING THE FOLDERS                                       ---- #
    # -------------------------------------------------------------------------------------------------------- #

    # prepare directories
    if data['bool_sub_dir']:
        data['model_dir'] = os.path.join(data['model_dir'], data['sub_dir'])
    os.makedirs(data['model_dir'], exist_ok=True)

    if data['log_dir']:
        if data['bool_sub_dir']:
            data['log_dir'] = os.path.join(data['log_dir'], data['sub_dir'])
        os.makedirs(data['log_dir'], exist_ok=True)

    # -------------------------------------------------------------------------------------------------------- #
    # ----                                   SETTING MODELS PARAMETERS                                     ---- #
    # -------------------------------------------------------------------------------------------------------- #

    # multi-GPU support
    context = contextlib.nullcontext()
    if nb_devices > 1:
        context = tf.distribute.MirroredStrategy().scope()

    # model configuration (grayscale images generation)
    gen_args = dict(
        in_shape=in_shape,
        in_label_list=labels_in,
        out_label_list=labels_out,
        warp_std=data['vel_std'],
        warp_res=data['vel_res'],
        blur_std=data['blur_std'],
        bias_std=data['bias_std'],
        bias_res=data['bias_res'],
        gamma_std=data['gamma'],
    )

    # model configuration (registration)
    reg_args = dict(
        inshape=in_shape,
        int_steps=data['int_steps'],
        int_resolution=data['int_res'],
        svf_resolution=data['svf_res'],
        nb_unet_features=(data['enc'], data['dec']),
    )

    # -------------------------------------------------------------------------------------------------------- #
    # ----            BUILDING THE MODELS FOR GRAYSCALE IMAGES GENERATION AND FOR REGISTRATION            ---- #
    # -------------------------------------------------------------------------------------------------------- #

    # build model
    with context:

        # generation of grayscale images
        gen_model_1 = ne.models.labels_to_image(**gen_args, id=0)
        gen_model_2 = ne.models.labels_to_image(**gen_args, id=1)
        ima_1, map_1 = gen_model_1.outputs
        ima_2, map_2 = gen_model_2.outputs

        # registration of the images
        inputs = gen_model_1.inputs + gen_model_2.inputs
        reg_args['input_model'] = tf.keras.Model(inputs, outputs=(ima_1, ima_2))
        model = vxm.networks.VxmDense(**reg_args)
        flow = model.references.pos_flow
        pred = vxm.layers.SpatialTransformer(interp_method='linear', name='pred')([map_1, flow])

        # losses and compilation
        model.add_loss(vxm.losses.Dice().loss(map_2, pred) + tf.repeat(1., data['batch_size']))
        model.add_loss(vxm.losses.Grad('l2', loss_mult=data['reg_param']).loss(None, flow))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=data['lr']))
        model.summary()

    # callbacks
    steps_per_epoch = len(label_maps_tr) // data['batch_size']
    save_name = os.path.join(data['model_dir'], '{epoch:04d}.h5')
    save = tf.keras.callbacks.ModelCheckpoint(
        save_name,
        save_freq=steps_per_epoch * data['save_freq'],
    )
    callbacks = [save]

    if data['log_dir']:
        log = tf.keras.callbacks.TensorBoard(
            log_dir=data['log_dir'],
            write_graph=False,
        )
        callbacks.append(log)

    # -------------------------------------------------------------------------------------------------------- #
    # ----                                INITIALIZE AND TRAIN THE MODEL                                  ---- #
    # -------------------------------------------------------------------------------------------------------- #

    # initialize and fit
    if data['bool_init_weights']:
        model.load_weights(data['init_weights'])
    model.save(save_name.format(epoch=data['init_epoch']))
    model.fit(
        gen_tr,
        validation_data=gen_val,
        validation_steps=len(label_maps_val) // data['batch_size_val'],
        initial_epoch=data['init_epoch'],
        epochs=data['epochs'],
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        verbose=data['verbose'],
    )
