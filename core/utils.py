import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

_CLASS_NAMES_FILE = './data/coco.names'

def load_class_names():
  """Returns a list of string corresonding to class names and it's length"""
  with open(_CLASS_NAMES_FILE, 'r') as f:
    class_names = f.read().splitlines()

  return class_names, len(class_names)

def load_image(img_path, input_size):
  """Loads image in a 4D array"""
  img = Image.open(img_path)
  img = img.resize(size=input_size)
  img = np.array(img, dtype=np.float32)
  img = np.expand_dims(img[:, :, :3], axis=0)
  return img

def draw_boxes(img_name, boxes_dict, class_names, input_size):
  """Draws detected boxes"""
  img = Image.open(img_name)
  draw = ImageDraw.Draw(img)
  font = ImageFont.truetype(font="./data/Roboto-Black.ttf", size=(img.size[0] + img.size[1]) // 100)
  resize_factor = (img.size[0] / input_size[0], img.size[1] / input_size[1])

  for cls in range(len(class_names)):
    boxes = boxes_dict[cls]

    if np.size(boxes) != 0:
      for box in boxes:
        xy, confidence = box[:4], box[4]
        xy = [xy[i] * resize_factor[i % 2] for i in range(4)]
        x0, y0 = xy[0], xy[1]
        thickness = (img.size[0] + img.size[1]) // 300

        for t in np.linspace(0, 1, thickness):
          xy[0], xy[1] = xy[0] + t, xy[1] + t
          xy[2], xy[3] = xy[2] - t, xy[3] - t
          draw.rectangle(xy, outline="blue")

        text = f"{class_names[cls]} {(confidence * 100):.1f}%"
        text_size = draw.textsize(text, font=font)
        draw.rectangle([x0, y0 - text_size[1], x0 + text_size[0], y0], fill="blue")
        draw.text((x0, y0 - text_size[1]), text, fill="black", font=font)

        print(text)

  rgb_img = img.convert('RGB')
  rgb_img.save('./detections/image_output.jpg')
  print("Image Saved at \"" + './detections/image_output.jpg' + "\"")
  rgb_img.show()

def draw_boxes_frame(frame, frame_size, boxes_dicts, class_names, input_size):
  """Draws detected boxes in a video frame"""
  boxes_dict = boxes_dicts[0]
  resize_factor = (frame_size[0] / input_size[1], frame_size[1] / input_size[0])
  for cls in range(len(class_names)):
    boxes = boxes_dict[cls]
    color = (0, 0, 255)
    if np.size(boxes) != 0:
      for box in boxes:
        xy = box[:4]
        xy = [int(xy[i] * resize_factor[i % 2]) for i in range(4)]
        cv2.rectangle(frame, (xy[0], xy[1]), (xy[2], xy[3]), color[::-1], 2)
        (test_width, text_height), baseline = cv2.getTextSize(class_names[cls],
                                                              cv2.FONT_HERSHEY_SIMPLEX,
                                                              0.75, 1)
        cv2.rectangle(frame,
                      (xy[0], xy[1]),
                      (xy[0] + test_width, xy[1] - text_height - baseline),
                      color[::-1],
                      thickness=cv2.FILLED)
        cv2.putText(frame, class_names[cls], (xy[0], xy[1] - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
