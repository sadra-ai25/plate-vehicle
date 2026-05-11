import os

from fastapi import FastAPI
from service import Inference

# define the app and the api variables
APP_NAME = os.getenv("APP_NAME", 'Car_Recognition')

APP_ROOT = os.getenv('APP_ROOT', '/plate')
HOST = os.getenv("HOST", "127.0.0.1")

# PORT_NUMBER = int(os.getenv('PORT_NUMBER', 3001))
PORT_NUMBER = int(os.getenv('PORT_NUMBER', 3002))
YOLO_PATH = os.getenv('YOLO_PATH', 'weights/best.pt')
RECOG_PATH = os.getenv('RECOG_PATH', 'weights/best.ckpt')
VEHICLE_PATH = os.getenv('VEHICLE_PATH', 'weights/vehicle.pt')
INPUT_SIZE = int(os.getenv('INPUT_SIZE', 160))

# load the model
inference = Inference(YOLO_PATH, RECOG_PATH, VEHICLE_PATH) 
app = FastAPI()