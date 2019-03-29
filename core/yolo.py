import tensorflow as tf

from core.layers import Conv2D, BatchNormalization, LeakyReLU, yolo_layer, upsample, build_boxes, non_max_suppression

_INPUT_SIZE = [416, 416]
_MAX_OUTPUT_SIZE = 20
_ANCHORS = [(10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)]

def darknet53_residual_block(inputs, filters, data_format, strides=1):
  """Creates a residual block for Darknet."""
  shortcut = inputs

  inputs = Conv2D(inputs, filters=filters, kernel_size=1, strides=strides, data_format=data_format)
  inputs = BatchNormalization(inputs, data_format=data_format)
  inputs = LeakyReLU(inputs)

  filters *= 2
  inputs = Conv2D(inputs, filters=filters, kernel_size=3, strides=strides, data_format=data_format)
  inputs = BatchNormalization(inputs, data_format=data_format)
  inputs = LeakyReLU(inputs)

  inputs += shortcut
  return inputs

def darknet53(inputs, data_format):
  """Creates Darknet53 model"""
  inputs = Conv2D(inputs, filters=32, kernel_size=3, data_format=data_format)
  inputs = BatchNormalization(inputs, data_format=data_format)
  inputs = LeakyReLU(inputs)
  inputs = Conv2D(inputs, filters=64, kernel_size=3, strides=2, data_format=data_format)
  inputs = BatchNormalization(inputs, data_format=data_format)
  inputs = LeakyReLU(inputs)

  inputs = darknet53_residual_block(inputs, filters=32, data_format=data_format)

  inputs = Conv2D(inputs, filters=128, kernel_size=3, strides=2, data_format=data_format)
  inputs = BatchNormalization(inputs, data_format=data_format)
  inputs = LeakyReLU(inputs)

  for _ in range(2):
    inputs = darknet53_residual_block(inputs, filters=64, data_format=data_format)

  inputs = Conv2D(inputs, filters=256, kernel_size=3, strides=2, data_format=data_format)
  inputs = BatchNormalization(inputs, data_format=data_format)
  inputs = LeakyReLU(inputs)

  for _ in range(8):
    inputs = darknet53_residual_block(inputs, filters=128, data_format=data_format)
  route4 = inputs # layers 36

  inputs = Conv2D(inputs, filters=512, kernel_size=3, strides=2, data_format=data_format)
  inputs = BatchNormalization(inputs, data_format=data_format)
  inputs = LeakyReLU(inputs)

  for _ in range(8):
    inputs = darknet53_residual_block(inputs, filters=256, data_format=data_format)
  route2 = inputs # layers 61

  inputs = Conv2D(inputs, filters=1024, kernel_size=3, strides=2, data_format=data_format)
  inputs = BatchNormalization(inputs, data_format=data_format)
  inputs = LeakyReLU(inputs)

  for _ in range(4):
    inputs = darknet53_residual_block(inputs, filters=512, data_format=data_format)
  return inputs, route2, route4

def feature_pyramid_network(inputs, filters, data_format):
  """Creates convolution operations layer used after Darknet"""
  inputs = Conv2D(inputs, filters=filters, kernel_size=1, data_format=data_format)
  inputs = BatchNormalization(inputs, data_format=data_format)
  inputs = LeakyReLU(inputs)

  inputs = Conv2D(inputs, filters=(filters * 2), kernel_size=3, data_format=data_format)
  inputs = BatchNormalization(inputs, data_format=data_format)
  inputs = LeakyReLU(inputs)

  inputs = Conv2D(inputs, filters=filters, kernel_size=1, data_format=data_format)
  inputs = BatchNormalization(inputs, data_format=data_format)
  inputs = LeakyReLU(inputs)

  inputs = Conv2D(inputs, filters=(filters * 2), kernel_size=3, data_format=data_format)
  inputs = BatchNormalization(inputs, data_format=data_format)
  inputs = LeakyReLU(inputs)

  inputs = Conv2D(inputs, filters=filters, kernel_size=1, data_format=data_format)
  inputs = BatchNormalization(inputs, data_format=data_format)
  inputs = LeakyReLU(inputs)
  route = inputs

  inputs = Conv2D(inputs, filters=(filters * 2), kernel_size=3, data_format=data_format)
  inputs = BatchNormalization(inputs, data_format=data_format)
  inputs = LeakyReLU(inputs)
  return inputs, route

class YOLOv3(object):
  """YOLOv3 model

  Parameters
  ----------
    n_classes: int
      Number of class labels
    input_size: list
      The input size of the model
    max_output_size: int
      Maximum number of boxes to be selected for each class
    iou_threshold: float
      Threshold for the IoU (Intersection over Unions)
    confidence_threshold: float
      Threshold for the confidence score
  """
  def __init__(self, n_classes, iou_threshold, confidence_threshold):

    self.n_classes = n_classes
    self.input_size = _INPUT_SIZE
    self.max_output_size = _MAX_OUTPUT_SIZE
    self.iou_threshold = iou_threshold
    self.confidence_threshold = confidence_threshold
    self.data_format = 'channels_first' if tf.test.is_built_with_cuda() else'channels_last'
    self.scope = 'yolov3'

  def __call__(self, inputs):
    """Generate Computation Graph"""
    with tf.variable_scope(self.scope):
      if self.data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

      inputs = inputs / 255

      inputs, route2, route4 = darknet53(inputs, data_format=self.data_format)

      inputs, route1 = feature_pyramid_network(inputs, filters=512, data_format=self.data_format)
      detect1 = yolo_layer(inputs,
                           n_classes=self.n_classes,
                           anchors=_ANCHORS[6:],
                           img_size=self.input_size,
                           data_format=self.data_format)

      inputs = Conv2D(route1, filters=256, kernel_size=1, data_format=self.data_format)
      inputs = BatchNormalization(inputs, data_format=self.data_format)
      inputs = LeakyReLU(inputs)

      upsample_size = route2.get_shape().as_list()
      inputs = upsample(inputs, out_shape=upsample_size, data_format=self.data_format)
      axis = 1 if self.data_format == 'channels_first' else 3
      inputs = tf.concat([inputs, route2], axis=axis)

      inputs, route3 = feature_pyramid_network(inputs, filters=256, data_format=self.data_format)
      detect2 = yolo_layer(inputs,
                           n_classes=self.n_classes,
                           anchors=_ANCHORS[3:6],
                           img_size=self.input_size,
                           data_format=self.data_format)

      inputs = Conv2D(route3, filters=128, kernel_size=1, data_format=self.data_format)
      inputs = BatchNormalization(inputs, data_format=self.data_format)
      inputs = LeakyReLU(inputs)

      upsample_size = route4.get_shape().as_list()
      inputs = upsample(inputs, out_shape=upsample_size, data_format=self.data_format)
      axis = 1 if self.data_format == 'channels_first' else 3
      inputs = tf.concat([inputs, route4], axis=axis)

      inputs, _ = feature_pyramid_network(inputs, filters=128, data_format=self.data_format)
      detect3 = yolo_layer(inputs,
                           n_classes=self.n_classes,
                           anchors=_ANCHORS[:3],
                           img_size=self.input_size,
                           data_format=self.data_format)

      inputs = tf.concat([detect1, detect2, detect3], axis=1)
      inputs = build_boxes(inputs)
      boxes_dicts = non_max_suppression(inputs,
                                        n_classes=self.n_classes,
                                        max_output_size=self.max_output_size,
                                        iou_threshold=self.iou_threshold,
                                        confidence_threshold=self.confidence_threshold)
      return boxes_dicts
