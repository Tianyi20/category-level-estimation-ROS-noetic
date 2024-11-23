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
from camera_realsense import initialize_camera,get_rgb
image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'pnp', 'track']


def demo(opt, meta):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    if opt.use_pnp == True and 'camera_matrix' not in meta.keys():
        raise RuntimeError('Error found. Please give the camera matrix when using pnp algorithm!')

    # here we initialize the camera
    pipeline = initialize_camera()
    #here must set     detector.pause = False
    detector.pause = False
    idx = 0
    while (True):

        img = get_rgb(pipeline)
        if img is None:
            # if the image fetch fail, directly go next loop
            continue
        try:
            cv2.imshow('input', img)
        except:
            exit(1)

        filename = os.path.splitext(os.path.basename(opt.demo))[0] + '_' + str(idx).zfill(
            4) + '.png'
        #print(filename) 这里的filename就是 webcam_000, web_cam111
        # main code: detect run
        ret = detector.run(img, meta_inp=meta,
                            filename=filename)
        print("ret result is",ret['results'])
        idx = idx + 1

        time_str = ''
        for stat in time_stats:
            time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(f'Frame {str(idx).zfill(4)}|' + time_str)
        # OpenCV must use along with CV.waitkey
        if cv2.waitKey(1) == 27:
            break



if __name__ == '__main__':

    # Default params with commandline input
    opt = opts().parser.parse_args()

    # Local machine configuration example for CenterPose
    # opt.c = 'cup' # Only meaningful when enables show_axes option
    # opt.demo = "../images/CenterPose/cup/00007.png"
    # opt.arch = 'dlav1_34'
    # opt.load_model = f"../models/CenterPose/cup_mug_v1_140.pth"
    # opt.debug = 2
    # opt.show_axes = True

    # Local machine configuration example for CenterPoseTrack
    # opt.c = 'cup' # Only meaningful when enables show_axes option
    # opt.demo = '../images/CenterPoseTrack/shoe_batch-25_10.mp4'
    # opt.tracking_task = True
    # opt.arch = 'dla_34'
    # opt.load_model = f"../models/CenterPoseTrack/shoe_15.pth"
    # opt.debug = 2
    # opt.show_axes = True

    # Default setting
    opt.nms = True
    opt.obj_scale = True
    opt.arch = "dlav1_34"
    opt.demo = "webcam"
    opt.load_model = "/home/tianyi/pose_estimation/src/CenterPose/models/cup_mug_v1_140.pth"
    # PnP related
    meta = {}
    
    # 相机内参
    meta['camera_matrix'] = np.array(
        [[615.0198364257812, 0.0, 317.4881591796875], 
         [0.0, 615.1835327148438, 244.118896484375], 
         [0.0, 0.0, 1.0]])      
    
    opt.cam_intrinsic = meta['camera_matrix']
    
    opt.use_pnp = True

    # Update default configurations
    opt = opts().parse(opt)

    # Update dataset info/training params
    opt = opts().init(opt)
    demo(opt, meta)
