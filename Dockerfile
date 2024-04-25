FROM nvidia/cuda:12.4.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0
COPY requirements.txt requirements.txt

COPY class-weights.npy class-weights.npy

COPY qAda.py qAda.py

RUN python3 -m pip install -r requirements.txt

RUN python3 -m pip install --upgrade pip

