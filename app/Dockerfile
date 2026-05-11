# FROM python:3.11.3-buster
FROM plate:latest

LABEL author="Ali Hadipour <a.hadipour64@gmail.com>"

ENV YOLO_DISABLE_AUTO_UPDATE=1

# RUN apt-get update -y \
#   && apt install -y libgl1-mesa-glx ffmpeg libsm6 libxext6 \
#   && python -m pip install --no-cache-dir --upgrade pip

# RUN pip install --no-cache-dir \
#     torch==2.0.1+cpu \
#     torchvision==0.15.2+cpu \

COPY . /app
WORKDIR /app

RUN git config --global --add safe.directory '*'
RUN pip install -r requirements.txt
