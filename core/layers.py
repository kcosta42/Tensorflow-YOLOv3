import tensorflow as tf

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1

def fixed_padding(inputs, kernel_size, data_format):
  """ResNet implementation of fixed padding"""
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

  return padded_inputs

def Conv2D(inputs, filters, kernel_size, data_format, strides=1):
  """Strided 2-D convolution with explicit padding"""
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.Conv2D(filters=filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding=('same' if strides == 1 else 'valid'),
                          use_bias=False,
                          data_format=data_format)(inputs)

def BatchNormalization(inputs, data_format):
  """Performs a batch normalization"""
  return tf.layers.BatchNormalization(axis=1 if data_format == 'channels_first' else 3,
                                      momentum=_BATCH_NORM_DECAY,
                                      epsilon=_BATCH_NORM_EPSILON,
                                      scale=True)(inputs)

def MaxPooling2D(inputs, pool_size, strides, data_format):
  """Performs a max pooling"""
  return tf.layers.MaxPooling2D(pool_size,
                                strides,
                                padding='same',
                                data_format=data_format)(inputs)

def LeakyReLU(inputs):
  return tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

def yolo_layer(inputs, n_classes, anchors, img_size, data_format):
  """Creates Yolo final detection layer"""
  n_anchors = len(anchors)
  inputs = tf.layers.Conv2D(filters=n_anchors * (5 + n_classes),
                            kernel_size=1,
                            strides=1,
                            use_bias=True,
                            data_format=data_format)(inputs)

  shape = inputs.get_shape().as_list()
  grid_shape = shape[2:4] if data_format == 'channels_first' else shape[1:3]
  strides = (img_size[0] // grid_shape[0], img_size[1] // grid_shape[1])

  if data_format == 'channels_first':
    inputs = tf.transpose(inputs, [0, 2, 3, 1])
  inputs = tf.reshape(inputs, [-1, n_anchors * grid_shape[0] * grid_shape[1], 5 + n_classes])

  box_centers, box_shapes, confidence, classes = tf.split(inputs, [2, 2, 1, n_classes], axis=-1)

  x = tf.range(grid_shape[0], dtype=tf.float32)
  y = tf.range(grid_shape[1], dtype=tf.float32)
  x_offset, y_offset = tf.meshgrid(x, y)
  x_offset = tf.reshape(x_offset, (-1, 1))
  y_offset = tf.reshape(y_offset, (-1, 1))
  x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
  x_y_offset = tf.tile(x_y_offset, [1, n_anchors])
  x_y_offset = tf.reshape(x_y_offset, [1, -1, 2])

  box_centers = tf.nn.sigmoid(box_centers)
  box_centers = (box_centers + x_y_offset) * strides
  anchors = tf.tile(anchors, [grid_shape[0] * grid_shape[1], 1])
  box_shapes = tf.exp(box_shapes) * tf.cast(anchors, tf.float32)
  confidence = tf.nn.sigmoid(confidence)
  classes = tf.nn.sigmoid(classes)

  inputs = tf.concat([box_centers, box_shapes, confidence, classes], axis=-1)
  return inputs

def upsample(inputs, out_shape, data_format):
  """Upsamples to out_shape using nearest neighbor interpolation"""
  if data_format == 'channels_first':
    inputs = tf.transpose(inputs, [0, 2, 3, 1])
    new_height = out_shape[3]
    new_width = out_shape[2]
  else:
    new_height = out_shape[2]
    new_width = out_shape[1]

  inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))
  if data_format == 'channels_first':
    inputs = tf.transpose(inputs, [0, 3, 1, 2])
  return inputs

def build_boxes(inputs):
  """Computes top left and bottom right points of the boxes"""
  center_x, center_y, width, height, confidence, classes = tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)

  top_left_x = center_x - width / 2
  top_left_y = center_y - height / 2
  bottom_right_x = center_x + width / 2
  bottom_right_y = center_y + height / 2
  return tf.concat([top_left_x, top_left_y, bottom_right_x, bottom_right_y, confidence, classes], axis=-1)

def non_max_suppression(inputs, n_classes, max_output_size, iou_threshold, confidence_threshold):
  """Performs non-max suppression separately for each class"""
  batch = tf.unstack(inputs)
  boxes_dicts = []

  for boxes in batch:
    boxes = tf.boolean_mask(boxes, boxes[:, 4] > confidence_threshold)
    classes = tf.argmax(boxes[:, 5:], axis=-1)
    classes = tf.expand_dims(tf.cast(classes, tf.float32), axis=-1)
    boxes = tf.concat([boxes[:, :5], classes], axis=-1)

    boxes_dict = dict()
    for cls in range(n_classes):
      mask = tf.equal(boxes[:, 5], cls)
      mask_shape = mask.get_shape()
      if mask_shape.ndims != 0:
        class_boxes = tf.boolean_mask(boxes, mask)
        boxes_coords, boxes_conf_scores, _ = tf.split(class_boxes, [4, 1, -1], axis=-1)
        boxes_conf_scores = tf.reshape(boxes_conf_scores, [-1])
        indices = tf.image.non_max_suppression(boxes_coords,
                                               boxes_conf_scores,
                                               max_output_size,
                                               iou_threshold)
        class_boxes = tf.gather(class_boxes, indices)
        boxes_dict[cls] = class_boxes[:, :5]

    boxes_dicts.append(boxes_dict)
  return boxes_dicts
