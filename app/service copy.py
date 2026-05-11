import cv2
import numpy as np
import os
import torch
from datetime import datetime
from boxes import Box
from crnn_recognition import CRNNInferenceTorch

class Inference:
    def __init__(self, yolo_path, recog_path):
        # self.detector = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_path)
        self.detector = torch.hub.load('yolov5', 'custom', source='local', path=yolo_path)
        self.recognizer = CRNNInferenceTorch(recog_path, device='cpu')

    # write a method for resizing the image
    @staticmethod
    def resize_image(img, width, height):
        resized_image = cv2.resize(img, (width, height))
        return resized_image

    @staticmethod
    def preprocessing(img, width, height) -> np.ndarray:
        if type(img) is not np.ndarray:
            img = np.array(img).astype(np.uint8)
        img = Inference.resize_image(img, width, height)
        img = img[..., ::-1]  # BGR to RGB
        return img

    def infer(self, image, width, height):
        img = self.preprocessing(image, width, height)
        plates = {}
        objects = self.detector(img)
        # if len(objects.names):
        boxes = objects.pandas().xyxy[0].values[:, :4]
        if len(boxes):
            # boxes = objects.pandas().xyxy[0].values[:, :4]
            print(boxes)
            results = []
            for box in boxes:
                img_part = Box.get_box_img(img, (box[1], box[0], box[3], box[2]))
                result = self.recognizer.infer(img_part)
                results.append(result)
            i = 1
            for res, box in zip(results, boxes):
                if len(res) == 8:
                    plate_data = {
                            "plate_part1": res[:2],
                            "plate_part2": res[2:3],
                            "plate_part3": res[3:6],
                            "plate_part4": res[6:]
                        }
                    plates[f"plate_{i}"] = plate_data
                    i += 1
                 
                    # save croped plate
                    ch = plate_data['plate_part1'] + plate_data['plate_part2'] + plate_data['plate_part3'] + plate_data['plate_part4'] + '.jpg'
                    crop_img_path = os.path.join('cropped-plate', ch)
                    img_part = cv2.cvtColor(img_part, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(crop_img_path, img_part)
                     # save orig plate
                    cv2.imwrite('original-plate/' + ch, image)

                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{'no_ocr'}_{timestamp}'.jpg'"
                    crop_img_path = os.path.join('cropped-plate', filename)
                    img_part = cv2.cvtColor(img_part, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(crop_img_path, img_part)                 
                
        else:
            plates["plate"] = {
                "plate_part1": None,
                "plate_part2": None,
                "plate_part3": None,
                "plate_part4": None
            }
            plates[f"plate"] = None
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{'no_object'}_{timestamp}'.jpg'"
            image_path = os.path.join('original-plate', filename)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_path, image)

        return plates
