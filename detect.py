import argparse
import tensorflow as tf
import cv2

from core.utils import load_class_names, load_image, draw_boxes, draw_boxes_frame
from core.yolo_tiny import YOLOv3_tiny
from core.yolo import YOLOv3

def main(mode, tiny, iou_threshold, confidence_threshold, path):
  class_names, n_classes = load_class_names()
  if tiny:
    model = YOLOv3_tiny(n_classes=n_classes,
                        iou_threshold=iou_threshold,
                        confidence_threshold=confidence_threshold)
  else:
    model = YOLOv3(n_classes=n_classes,
                   iou_threshold=iou_threshold,
                   confidence_threshold=confidence_threshold)
  inputs = tf.placeholder(tf.float32, [1, *model.input_size, 3])
  detections = model(inputs)
  saver = tf.train.Saver(tf.global_variables(scope=model.scope))

  with tf.Session() as sess:
    saver.restore(sess, './weights/model-tiny.ckpt' if tiny else './weights/model.ckpt')

    if mode == 'image':
      image = load_image(path, input_size=model.input_size)
      result = sess.run(detections, feed_dict={inputs: image})
      draw_boxes(path, boxes_dict=result[0], class_names=class_names, input_size=model.input_size)
      return

    cv2.namedWindow("Detections")
    video = cv2.VideoCapture(path)
    fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter('./detections/video_output.mp4', fourcc, fps, frame_size)
    print("Video being saved at \"" + './detections/video_output.mp4' + "\"")
    print("Press 'q' to quit")
    while True:
      retval, frame = video.read()
      if not retval:
        break
      resized_frame = cv2.resize(frame, dsize=tuple((x) for x in model.input_size[::-1]), interpolation=cv2.INTER_NEAREST)
      result = sess.run(detections, feed_dict={inputs: [resized_frame]})
      draw_boxes_frame(frame, frame_size, result, class_names, model.input_size)
      cv2.imshow("Detections", frame)
      key = cv2.waitKey(1) & 0xFF
      if key == ord('q'):
          break
      out.write(frame)
    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--tiny", action="store_true", help="enable tiny model")
  parser.add_argument("mode", choices=["video", "image"], help="detection mode")
  parser.add_argument("iou", metavar="iou", type=float, help="IoU threshold [0.0, 1.0]")
  parser.add_argument("confidence", metavar="confidence", type=float, help="confidence threshold [0.0, 1.0]")
  parser.add_argument("path", type=str, help="path to file")

  args = parser.parse_args()
  main(args.mode, args.tiny, args.iou, args.confidence, args.path)
