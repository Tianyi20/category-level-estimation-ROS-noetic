# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
print(sys.executable)
import os
import cv2

from lib.opts import opts
from lib.detectors.detector_factory import detector_factory
import glob
import numpy as np
from camera_realsense import initialize_camera, get_rgb

import copy
image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'pnp', 'track']

def demo_multiple(opt_multiple, meta):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt_multiple[0].gpus_str

    # 创建多个 detector 实例
    detectors = []
    for opt in opt_multiple:
        Detector = detector_factory[opt.task]
        detector = Detector(opt)
        detectors.append(detector)

    # 初始化相机
    pipeline = initialize_camera()
    
    idx = 0
    while True:
        img = get_rgb(pipeline)
        if img is None:
            # 如果图像获取失败，直接跳到下一次循环
            continue
        try:
            cv2.imshow('input', img)
        except:
            exit(1)

        # 使用每个模型进行推理
        for i, detector in enumerate(detectors):
            filename = f"model_{i}_frame_{str(idx).zfill(4)}.png"
            ret = detector.run(img, meta_inp=meta, filename=filename)
            print(f"Model {i} results:", ret['results'])

            # time_str = ''
            # for stat in time_stats:
            #     time_str += '{} {:.3f}s |'.format(stat, ret[stat])
            # print(f'Model {i} Frame {str(idx).zfill(4)} |' + time_str)

        idx += 1

        # OpenCV 必须使用 cv2.waitKey 以便显示图像
        if cv2.waitKey(1) == 27:  # 按下 'Esc' 键退出
            break

if __name__ == '__main__':
    # Default params with commandline input
    opt_1 = copy.deepcopy(opts().parser.parse_args())
    opt_1.debug = 5

    # 配置第一个模型
    opt_1.nms = True
    opt_1.obj_scale = True
    opt_1.arch = "dlav1_34"
    opt_1.load_model = "/home/tianyi/pose_estimation/src/CenterPose/models/cup_mug_v1_140.pth"
    
    # PnP 相关配置
    meta = {}
    meta['camera_matrix'] = np.array(
        [[615.0198364257812, 0.0, 317.4881591796875], 
         [0.0, 615.1835327148438, 244.118896484375], 
         [0.0, 0.0, 1.0]])      
    
    opt_1.cam_intrinsic = meta['camera_matrix']
    opt_1.use_pnp = True

    # 更新默认配置
    opt_1 = opts().parse(opt_1)
    opt_1 = opts().init(opt_1)

    # 创建第二个模型的配置
    opt_2 = copy.deepcopy(opt_1)
    opt_2.load_model = "/home/tianyi/pose_estimation/src/CenterPose/models/bottle_v1_sym_12_140.pth"

    # 将两个配置添加到列表中
    opt_multip = [opt_1, opt_2]
    
    # 使用多个模型进行推理
    demo_multiple(opt_multip, meta)
