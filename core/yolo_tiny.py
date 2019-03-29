import tensorflow as tf

from core.layers import Conv2D, BatchNormalization, MaxPooling2D, LeakyReLU, yolo_layer, upsample, build_boxes, non_max_suppression

_INPUT_SIZE = [416, 416]
_MAX_OUTPUT_SIZE = 20
_ANCHORS = [[10, 14], [23,   27], [37,  58],
            [81, 82], [135, 169], [344, 319]]

def darknet(inputs, data_format):
  """Creates Darknet model"""
  filters = 16
  for _ in range(4):
    inputs = Conv2D(inputs, filters, kernel_size=3, data_format=data_format)
    inputs = BatchNormalization(inputs, data_format=data_format)
    inputs = LeakyReLU(inputs)
    inputs = MaxPooling2D(inputs, pool_size=[2, 2], strides=[2, 2], data_format=data_format)
    filters *= 2

  inputs = Conv2D(inputs, filters=256, kernel_size=3, data_format=data_format)
  inputs = BatchNormalization(inputs, data_format=data_format)
  inputs = LeakyReLU(inputs)
  route = inputs # layers 8
  inputs = MaxPooling2D(inputs, pool_size=[2, 2], strides=[2, 2], data_format=data_format)

  inputs = Conv2D(inputs, filters=512, kernel_size=3, data_format=data_format)
  inputs = BatchNormalization(inputs, data_format=data_format)
  inputs = LeakyReLU(inputs)
  inputs = MaxPooling2D(inputs, pool_size=[2, 2], strides=[1, 1], data_format=data_format)

  inputs = Conv2D(inputs, filters=1024, kernel_size=3, data_format=data_format)
  inputs = BatchNormalization(inputs, data_format=data_format)
  inputs = LeakyReLU(inputs)
  return inputs, route

def feature_pyramid_network(inputs, data_format):
  """Creates convolution operations layer used after Darknet"""
  inputs = Conv2D(inputs, filters=256, kernel_size=1, data_format=data_format)
  inputs = BatchNormalization(inputs, data_format=data_format)
  inputs = LeakyReLU(inputs)
  route = inputs

  inputs = Conv2D(inputs, filters=512, kernel_size=3, data_format=data_format)
  inputs = BatchNormalization(inputs, data_format=data_format)
  inputs = LeakyReLU(inputs)
  return inputs, route

class YOLOv3_tiny(object):
  """YOLOv3 tiny model

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
    self.scope = 'yolov3_tiny'

  def __call__(self, inputs):
    """Generate Computation Graph"""
    with tf.variable_scope(self.scope):
      if self.data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

      inputs = inputs / 255

      inputs, route2 = darknet(inputs, data_format=self.data_format)
      inputs, route1 = feature_pyramid_network(inputs, data_format=self.data_format)
      detect1 = yolo_layer(inputs,
                           n_classes=self.n_classes,
                           anchors=_ANCHORS[3:],
                           img_size=self.input_size,
                           data_format=self.data_format)

      inputs = Conv2D(route1, filters=128, kernel_size=1, data_format=self.data_format)
      inputs = BatchNormalization(inputs, data_format=self.data_format)
      inputs = LeakyReLU(inputs)

      upsample_size = route2.get_shape().as_list()
      inputs = upsample(inputs, out_shape=upsample_size, data_format=self.data_format)
      axis = 1 if self.data_format == 'channels_first' else 3
      inputs = tf.concat([inputs, route2], axis=axis)

      inputs = Conv2D(inputs, filters=256, kernel_size=3, data_format=self.data_format)
      inputs = BatchNormalization(inputs, data_format=self.data_format)
      inputs = LeakyReLU(inputs)
      detect2 = yolo_layer(inputs,
                           n_classes=self.n_classes,
                           anchors=_ANCHORS[:3],
                           img_size=self.input_size,
                           data_format=self.data_format)

      inputs = tf.concat([detect1, detect2], axis=1)
      inputs = build_boxes(inputs)
      boxes_dicts = non_max_suppression(inputs,
                                        n_classes=self.n_classes,
                                        max_output_size=self.max_output_size,
                                        iou_threshold=self.iou_threshold,
                                        confidence_threshold=self.confidence_threshold)
      return boxes_dicts
