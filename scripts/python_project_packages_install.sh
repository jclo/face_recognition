#!/bin/sh
# ******************************************************************************
# A bash script to install Python modules in the project environment required
# by the project.
# ******************************************************************************
pip install --upgrade \
  opencv-python \
  numpy \
  mtcnn \
  dlib \
  matplotlib \
  keras \
  tensorflow \
  pillow
