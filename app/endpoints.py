from typing import Union

import numpy as np
from fastapi import Request

from settings import app, APP_ROOT, inference

def b64_to_img(image_byte: Union[bytes, str]) -> np.ndarray:
    """
    Converts the input bytes or string to an 3-channel image
    :param image_byte: base64 image string
    :return: numpy image
    """
    import base64
    import cv2
    if isinstance(image_byte, bytes):
        image_byte = image_byte.decode()
    if ";" in image_byte:
        image_byte = image_byte.split(";")[-1]

    image_byte = image_byte.encode()
    im_bytes = base64.b64decode(image_byte)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

    return img

@app.post(APP_ROOT)
async def plate_recognition(request: Request):
    r = await request.json()
    image = r['image']
    side = r.get('side', None)
    img = b64_to_img(image)
    res = inference.infer(img, width=1920, height=1080, side=side)
    return res

@app.get(APP_ROOT)
def get_plate():
    return {"Just": "Fine!"}