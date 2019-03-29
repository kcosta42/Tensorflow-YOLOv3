import argparse
import tensorflow as tf
import numpy as np

from core.yolo_tiny import YOLOv3_tiny
from core.yolo import YOLOv3

"""Instructions for converting weights:

The first 5 values correspond to
major version (4 bytes)
minor version (4 bytes)
revision      (4 bytes)
images seen   (8 bytes)

Darknet store Kernels in Caffe-Style          : [out_channels, in_channels, height, width]
We need to transpose them in Tensorflow-Style : [height, width, in_channels, out_channels]
"""

def load_batch_norm(idx, variables, weights, assign_ops, offset):
  """Loads kernel, gamma, beta, mean, variance for Batch Normalization"""
  kernel = variables[idx]
  gamma, beta, mean, variance = variables[idx + 1:idx + 5]
  batch_norm_vars = [beta, gamma, mean, variance]

  for var in batch_norm_vars:
    shape = var.shape.as_list()
    num_params = np.prod(shape)
    var_weights = weights[offset:offset + num_params].reshape(shape)
    offset += num_params
    assign_ops.append(tf.assign(var, var_weights))

  shape = kernel.shape.as_list()
  num_params = np.prod(shape)
  var_weights = weights[offset:offset + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
  var_weights = np.transpose(var_weights, (2, 3, 1, 0))
  offset += num_params
  assign_ops.append(tf.assign(kernel, var_weights))
  return assign_ops, offset

def load_weights(variables, filename):
  """Loads official pretrained YOLOv3 weights"""
  with open(filename, "rb") as f:
    print("Loading weights from \"" + filename + "\"")

    # Skip first 5 values
    np.fromfile(f, dtype=np.int32, count=5)
    weights = np.fromfile(f, dtype=np.float32)

    assign_ops = []
    offset = 0

    # Load weights for Darknet part.
    # Each convolution layer has batch normalization.
    for i in range(52):
      idx = 5 * i
      assign_ops, offset = load_batch_norm(idx, variables, weights, assign_ops, offset)

    # Loading weights for Yolo part.
    # 7th, 15th and 23rd convolution layer has biases and no batch norm.
    ranges = [range(0, 6), range(6, 13), range(13, 20)]
    unnormalized = [6, 13, 20]
    for j in range(3):
      for i in ranges[j]:
        idx = 52 * 5 + 5 * i + j * 2
        assign_ops, offset = load_batch_norm(idx, variables, weights, assign_ops, offset)

      bias = variables[52 * 5 + unnormalized[j] * 5 + j * 2 + 1]
      shape = bias.shape.as_list()
      num_params = np.prod(shape)
      var_weights = weights[offset:offset + num_params].reshape(shape)
      offset += num_params
      assign_ops.append(tf.assign(bias, var_weights))

      kernel = variables[52 * 5 + unnormalized[j] * 5 + j * 2]
      shape = kernel.shape.as_list()
      num_params = np.prod(shape)
      var_weights = weights[offset:offset + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
      var_weights = np.transpose(var_weights, (2, 3, 1, 0))
      offset += num_params
      assign_ops.append(tf.assign(kernel, var_weights))
  return assign_ops

def load_weights_tiny(variables, filename):
  """Loads official pretrained YOLOv3-tiny weights"""
  with open(filename, "rb") as f:
    print("Loading weights from \"" + filename + "\"")

    # Skip 5 first values
    _ = np.fromfile(f, dtype=np.int32, count=5)
    weights = np.fromfile(f, dtype=np.float32)

    assign_ops = []
    offset = 0

    # Load weights for Darknet part.
    # Each convolution layer has batch normalization.
    for i in range(7):
      idx = 5 * i
      assign_ops, offset = load_batch_norm(idx, variables, weights, assign_ops, offset)

    # Loading weights for Yolo part.
    # 3rd and 5th convolution layer has biases and no batch norm.
    ranges = [range(0, 2), range(2, 4)]
    unnormalized = [2, 4]
    for j in range(2):
      for i in ranges[j]:
        idx = 7 * 5 + 5 * i + j * 2
        assign_ops, offset = load_batch_norm(idx, variables, weights, assign_ops, offset)

      bias = variables[7 * 5 + unnormalized[j] * 5 + j * 2 + 1]
      shape = bias.shape.as_list()
      num_params = np.prod(shape)
      var_weights = weights[offset:offset + num_params].reshape(shape)
      offset += num_params
      assign_ops.append(tf.assign(bias, var_weights))

      kernel = variables[7 * 5 + unnormalized[j] * 5 + j * 2]
      shape = kernel.shape.as_list()
      num_params = np.prod(shape)
      var_weights = weights[offset:offset + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
      var_weights = np.transpose(var_weights, (2, 3, 1, 0))
      offset += num_params
      assign_ops.append(tf.assign(kernel, var_weights))
  return assign_ops

def main(tiny):
  if tiny:
    model = YOLOv3_tiny(n_classes=80,
                        iou_threshold=0.5,
                        confidence_threshold=0.5)
  else:
    model = YOLOv3(n_classes=80,
                   iou_threshold=0.5,
                   confidence_threshold=0.5)

  inputs = tf.placeholder(tf.float32, [1, 416, 416, 3])
  model(inputs)
  model_vars = tf.global_variables(scope=model.scope)
  if tiny:
    assign_ops = load_weights_tiny(model_vars, './weights/yolov3-tiny.weights')
  else:
    assign_ops = load_weights(model_vars, './weights/yolov3.weights')

  saver = tf.train.Saver(tf.global_variables(scope=model.scope))
  with tf.Session() as sess:
    save_path = './weights/model-tiny.ckpt' if tiny else './weights/model.ckpt'
    sess.run(assign_ops)
    saver.save(sess, save_path)
    print("Model Saved at \"" + save_path + "\"")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--tiny", action="store_true", help="enable tiny model")

  args = parser.parse_args()
  main(args.tiny)
