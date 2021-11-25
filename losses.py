"""
File including the custom losses (losses differing from the ones used in voxelmorph)
The custom losses included are:
    - dice_loss_zeropad(y_true, y_pred): this loss is similar to a dice loss, except that it ignores the
      zero-padded parts of the label maps. It is used to train the registration model and replaces the normal dice loss
      when zero-padding is applied either to the training data or to the validation data.
"""

import tensorflow as tf

def dice_loss_zeropad(y_true, y_pred):
    """
    Dice loss computed only on regions where no zero-padding was done considering both y_true and y_pred.
    Dice loss to be applied on label maps of shape [None, x, y, z, n_labels].
    Process:
    1. Identify all the voxels associated to the label 0 in the source and target images (take the subvolume
    corresponding to the label 0 and identify all the voxels with a value greater or equal to 1)
    2. Compute the Dice score for all the other labels, ignoring the voxels identified in point 1 (for each subvolume
    corresponding to a certain label, set the voxels identified in 1 to 0 in both the source and the target subvolumes)
    3. Take the mean of the Dice score of each label (except the label 0)
    4. Return the Dice loss (-dice score)
    """

    ndims = len(y_pred.get_shape().as_list()) - 2
    vol_axes = list(range(1, ndims + 1))

    if ndims != 3:
        err = f"The Dice loss computed only on regions with no zero-padding can only be used on 3D volumes " \
              f"but the dimension of the object is: {ndims}. The expected input should be of shape " \
              f"[None, x, y, z, n_labels] but received: " \
              f"{y_true.get_shape().as_list()} and {y_pred.get_shape().as_list()}"
    raise ValueError(err)

    # Create a map of the shape of interest that will be used to compare with the subvolume representing the 0 label
    map = tf.constant(1, shape=y_pred.get_shape().as_list()[1:-1], dtype=tf.float32)

    # Determine the zero-padded areas by comparing the subvolume representing the 0 label with the map
    is_y_true_0 = tf.greater_equal(y_true[0, :, :, :, 0], map)
    is_y_pred_0 = tf.greater_equal(y_pred[0, :, :, :, 0], map)

    # Create a mask that represents where there was no zero-padding (is_not_0_element)
    is_0_element = tf.math.logical_or(is_y_true_0, is_y_pred_0)
    is_not_0_element = tf.math.logical_not(is_0_element)

    # Create a map of 0 to replace the values of the different labels with 0 at the place where 0 padding has been added
    zero_map = tf.constant(0, shape=y_pred.get_shape().as_list()[1:-1])
    zero_map = tf.cast(zero_map, tf.float32)

    # for each label, set the values to 0 if it's in an area where zero-padding was applied
    y_true_list = []
    y_pred_list = []
    for i in range(y_pred.get_shape().as_list()[-1]):
        y_true_list.append(tf.where(is_not_0_element, y_true[0, :, :, :, i], zero_map))
        y_pred_list.append(tf.where(is_not_0_element, y_pred[0, :, :, :, i], zero_map))

    y_true = tf.stack(y_true_list)
    y_pred = tf.stack(y_pred_list)

    top = 2 * tf.reduce_sum(y_true * y_pred, vol_axes)
    bottom = tf.reduce_sum(y_true + y_pred, vol_axes)

    # Remove the value computed on the subvolume corresponding to the 0 label
    top_non_zero_label = top[1:]
    bottom_non_zero_label = bottom[1:]

    div_no_nan = tf.math.divide_no_nan
    dice = tf.reduce_mean(div_no_nan(top_non_zero_label, bottom_non_zero_label))

    return -dice