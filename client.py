import os

import cv2
import requests
import numpy as np
from typing import Union

# Server
ENDPOINT_URL = "http://127.0.0.1:3001/plate"
image_path = '/home/ai/Downloads/images/Iran-Khodro-Dena.jpg'


def show_destroy_cv2(img, win_name="", show=True):
    if show:
        try:
            cv2.imshow(win_name, img)
            cv2.waitKey(0)
            cv2.destroyWindow(win_name)
        except Exception as e:
            cv2.destroyWindow(win_name)
            raise e


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


def img_to_b64(image: np.ndarray, extension: str = ".jpg") -> str:
    """
    returns a base64 encoded string from an image
    :param image: The input image
    :param extension: the extension to encode
    :return: byte string
    """
    import base64

    import cv2

    _, encoded_img = cv2.imencode(extension, image)
    base64_img = base64.b64encode(encoded_img).decode("utf-8")
    return base64_img


def infer():
    image = cv2.imread(image_path)[..., ::-1]
    byte_img = img_to_b64(image)
    data = {'image': byte_img}
    response = requests.post(ENDPOINT_URL, json=data)
    response.raise_for_status()
    print(response.json())


def _get_file_content(file_path: str, file_content_type: str = 'multipart/form-data') -> tuple:
    file_name = os.path.basename(file_path)
    file_content = open(file_path, 'rb')
    return file_name, file_content, file_content_type


if __name__ == "__main__":
    infer()