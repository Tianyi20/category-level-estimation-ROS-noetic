# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2

from lib.opts import opts
from lib.detectors.detector_factory import detector_factory
import glob
import numpy as np
from camera_realsense import initialize_camera, get_rgb



pipeline = initialize_camera()

try:
    while True:
        img = get_rgb(pipeline)
        if img is None:
            print("No image received. Skipping...")
            continue

        print(f"Image shape: {img.shape}")  # 检查图像形状
        cv2.imshow('input', img)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
