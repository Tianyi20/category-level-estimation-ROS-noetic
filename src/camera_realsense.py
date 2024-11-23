## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import sys

def initialize_camera():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    return pipeline

def get_rgb(pipeline):
    try:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        #depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None

        # Convert images to numpy arrays
        #depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return color_image
    
    except Exception as e:
        print(f"Error retrieving frames: {e}")
        return None
    
 
if __name__ == "__main__":
    print("Starting camera pipeline...")
    
    pipeline = initialize_camera()
    if pipeline is None:
        print("Failed to initialize the camera.")
        sys.exit(1)

    print("Camera initialized. Fetching frames...")
    for counter in range(100):  # 试图抓取10帧作为示例
        rgb = get_rgb(pipeline)
        print(f"Frame shape: {rgb.shape}, dtype: {rgb.dtype}")

        if rgb is None:
            print(f"Failed to retrieve frame {counter}")
            continue

        print(f"Frame {counter}: RGB shape={rgb.shape}")
        # 保存图像到zarr文件...
    
    pipeline.stop()


    