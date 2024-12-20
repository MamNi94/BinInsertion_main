import threading
import queue
import time

import numpy as np
import pyrealsense2 as rs

import cv2


def capture_frames(pipeline,frame_queue,align, temporal_filter=None, spatial_filter=None, hole_filling_filter=None):

    

    while True:
        
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if temporal_filter is not None:
            filtered_depth_frame = temporal_filter.process(depth_frame) 
            filtered_depth_frame = spatial_filter.process(filtered_depth_frame)
            filtered_depth_frame = hole_filling_filter.process(filtered_depth_frame)

        if not color_frame or not depth_frame:
            continue
        
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(filtered_depth_frame.get_data())
    
        # Put the latest frame into the queue, discard older frames
        if not frame_queue.full():
            frame_queue.put((color_image, depth_image))
        else:
            try:
                frame_queue.get_nowait()  # Discard the old frame
            except queue.Empty:
                pass
            frame_queue.put((color_image, depth_image))