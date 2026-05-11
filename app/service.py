import cv2
import numpy as np
import os
import torch
from datetime import datetime
from boxes import Box
from crnn_recognition import CRNNInferenceTorch

class Inference:
    def __init__(self, yolo_path, recog_path, vehicle_path):
        self.detector = torch.hub.load('yolov5', 'custom', source='local', path=yolo_path)
        self.recognizer = CRNNInferenceTorch(recog_path, device='cpu')
        self.vehicle_detector = torch.hub.load('yolov5', 'custom', source='local', path=vehicle_path)
        
        os.makedirs('cropped-plate', exist_ok=True)
        # os.makedirs('original-plate', exist_ok=True)

    @staticmethod
    def resize_image(img, width, height):
        return cv2.resize(img, (width, height))

    @staticmethod
    def preprocessing(img, width, height) -> np.ndarray:
        if type(img) is not np.ndarray:
            img = np.array(img).astype(np.uint8)
        img = Inference.resize_image(img, width, height)
        img = img[..., ::-1]
        return img

    @staticmethod
    def get_center(box):
        """محاسبه مرکز باکس"""
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    @staticmethod
    def is_plate_inside_vehicle(plate_box, vehicle_box):
        """بررسی اینکه آیا پلاک داخل خودرو است"""
        plate_center = Inference.get_center(plate_box)
        # بررسی اینکه مرکز پلاک داخل باکس خودرو باشد
        return (vehicle_box[0] <= plate_center[0] <= vehicle_box[2] and 
                vehicle_box[1] <= plate_center[1] <= vehicle_box[3])

    def detect_vehicles(self, img, side=None):
        """
        تشخیص خودروها
        side: 'R' برای راست‌ترین، 'L' برای چپ‌ترین، None برای همه
        """
        vehicles = self.vehicle_detector(img)
        if len(vehicles.pandas().xyxy[0]) == 0:
            return []
            
        vehicle_df = vehicles.pandas().xyxy[0]
        boxes = vehicle_df.values[:, :4]
        classes = vehicle_df['name'].values
        
        vehicle_list = []
        for i in range(len(boxes)):
            vehicle_list.append({
                "type": str(classes[i]),
                "bbox": boxes[i].tolist(),
                "x1": boxes[i][0]  # برای مرتب‌سازی
            })
        
        if side == 'R':
            # راست‌ترین: بیشترین x1
            vehicle_list.sort(key=lambda x: x['x1'], reverse=True)
            return [vehicle_list[0]] if vehicle_list else []
        elif side == 'L':
            # چپ‌ترین: کمترین x1
            vehicle_list.sort(key=lambda x: x['x1'])
            return [vehicle_list[0]] if vehicle_list else []
        else:
            # همه خودروها، مرتب شده از چپ به راست
            vehicle_list.sort(key=lambda x: x['x1'])
            return vehicle_list

    def detect_plates(self, img):
        """تشخیص پلاک‌ها"""
        objects = self.detector(img)
        plates = []
        
        if len(objects.pandas().xyxy[0]) > 0:
            boxes = objects.pandas().xyxy[0].values[:, :4]
            for box in boxes:
                plates.append({
                    "bbox": box.tolist()
                })
        return plates

    def recognize_plate(self, img, plate_box):
        """تشخیص متن پلاک"""
        img_part = Box.get_box_img(img, (plate_box[1], plate_box[0], plate_box[3], plate_box[2]))
        result = self.recognizer.infer(img_part)
        return result, img_part

    def infer(self, image, width, height, side=None):
        img = self.preprocessing(image, width, height)
        
        # تشخیص خودروها
        vehicles = self.detect_vehicles(img, side)
        
        # تشخیص پلاک‌ها
        plates_detected = self.detect_plates(img)
        
        # تطابق پلاک‌ها با خودروها
        results = []
        
        for vehicle in vehicles:
            vehicle_box = vehicle["bbox"]
            matched_plate = None
            plate_img = None
            
            # یافتن پلاک مربوط به این خودرو
            for plate in plates_detected:
                if self.is_plate_inside_vehicle(plate["bbox"], vehicle_box):
                    # تشخیص متن پلاک
                    plate_text, plate_img = self.recognize_plate(img, plate["bbox"])
                    if len(plate_text) == 8:
                        matched_plate = {
                            "plate_part1": plate_text[:2],
                            "plate_part2": plate_text[2:3],
                            "plate_part3": plate_text[3:6],
                            "plate_part4": plate_text[6:],
                            "bbox": plate["bbox"]
                        }
                    break  # فقط یک پلاک برای هر خودرو
            
            # ساخت نتیجه
            if matched_plate:
                plate_data = {
                    "plate_part1": matched_plate["plate_part1"],
                    "plate_part2": matched_plate["plate_part2"],
                    "plate_part3": matched_plate["plate_part3"],
                    "plate_part4": matched_plate["plate_part4"],
                    "type": vehicle["type"]
                }
                
                # ذخیره تصویر پلاک
                ch = (matched_plate["plate_part1"] + matched_plate["plate_part2"] + 
                      matched_plate["plate_part3"] + matched_plate["plate_part4"] + '.jpg')
                crop_img_path = os.path.join('cropped-plate', ch)
                img_part_bgr = cv2.cvtColor(plate_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(crop_img_path, img_part_bgr)
                # cv2.imwrite('original-plate/' + ch, image)
                
            else:
                # خودرو بدون پلاک تشخیص داده شده
                plate_data = {
                    "plate_part1": "",
                    "plate_part2": "",
                    "plate_part3": "",
                    "plate_part4": "",
                    "type": vehicle["type"]
                }
                
                # ذخیره تصویر خودرو بدون پلاک
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"no_plate_{vehicle['type']}_{timestamp}.jpg"
                vehicle_img = Box.get_box_img(img, (vehicle_box[1], vehicle_box[0], vehicle_box[3], vehicle_box[2]))
                vehicle_img_bgr = cv2.cvtColor(vehicle_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join('cropped-plate', filename), vehicle_img_bgr)
            
            results.append(plate_data)
        
        # اگر هیچ خودرویی تشخیص داده نشد
        if not vehicles:
            # بررسی پلاک‌های یتیم (بدون خودرو)
            if plates_detected:
                for i, plate in enumerate(plates_detected):
                    plate_text, plate_img = self.recognize_plate(img, plate["bbox"])
                    if len(plate_text) == 8:
                        plate_data = {
                            "plate_part1": plate_text[:2],
                            "plate_part2": plate_text[2:3],
                            "plate_part3": plate_text[3:6],
                            "plate_part4": plate_text[6:],
                            "type": None
                        }
                        results.append(plate_data)
                        
                        # ذخیره
                        ch = plate_data["plate_part1"] + plate_data["plate_part2"] + plate_data["plate_part3"] + plate_data["plate_part4"] + '.jpg'
                        crop_img_path = os.path.join('cropped-plate', ch)
                        img_part_bgr = cv2.cvtColor(plate_img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(crop_img_path, img_part_bgr)
                        # cv2.imwrite('original-plate/' + ch, image)
            else:
                # هیچ چیزی تشخیص داده نشد
                results.append({
                    "plate_part1": "",
                    "plate_part2": "",
                    "plate_part3": "",
                    "plate_part4": "",
                    "type": None
                })
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"no_object_{timestamp}.jpg"
                # cv2.imwrite(os.path.join('original-plate', filename), image)

        # ساخت خروجی نهایی
        plates_dict = {}
        for idx, res in enumerate(results, 1):
            plates_dict[f"plate_{idx}"] = res
        
        response = {"plates": plates_dict}
        
        if side and vehicles:
            response["side"] = side
            
        return response