import numpy as np
import pyrealsense2 as rs
import cv2
import os
import heapq

from tensorflow.keras.preprocessing import image
import tensorflow as tf

from video_capture import capture_frames
from roi_functions import cut_region_at_distance, detect_roi,cut_region_v2, get_shifted_points,get_corner_points, draw_rotated_rectangle, cut_region_between_hulls
import threading
import queue
import time

ht = 0.5

wall_model = tf.keras.models.load_model('models\walls\inception_wall_rect_224x224_v0_L2_val_accuracy_0.993_combined_data.h5')


img_shape = 224

hole_model = tf.keras.models.load_model('models\holes\inception_hole_224x224_close.h5')
 
 # Only store the most recent frame
    
def detect_holes(rect,img, hole_model, img_shape = 224,hole_threshhold = 0.5):
    
    
    rect = np.array(rect,dtype = np.float32)
    width = np.linalg.norm(rect[0] - rect[1])
    height = np.linalg.norm(rect[0] - rect[3])

    dst_pts = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype="float32")

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst_pts)

    # Perform the perspective transformation (i.e., crop the image)
    cropped_image = cv2.warpPerspective(img, M, (int(width), int(height)))
    
    ##Save IMAGE
    
    
    cropped_image = cv2.resize(cropped_image,None,fx = img_shape/cropped_image.shape[1],fy = img_shape/cropped_image.shape[0])
    
    inspect_holes = True
    if cropped_image.shape[0] !=img_shape or cropped_image.shape[1]  != img_shape:
        inspect_holes = False
    #model
    if inspect_holes == True:
        img_array = image.img_to_array(cropped_image)
        
        img_array = np.expand_dims(img_array, axis=0)  # Create batch axis
        img_array /= 255.0  # Normalize
        prediction = hole_model.predict(img_array)
        print('hole prediction',prediction[0])
        text_offset_x = 150
        text_offset_y = 80
        if prediction[0] > hole_threshhold:
            cv2.putText(img,f'Passed', (int(rect[0][0]-text_offset_x),int(rect[0][1] + text_offset_y)), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 0), 2)
        else:
            cv2.putText(img,f'Failed', (int(rect[0][0]-text_offset_x),int(rect[0][1] + text_offset_y)), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 200), 2)
  
def detect_walls(color_image,masked_color_image, wall_model, number =1):
    t = time.time()
    factor_x = 224/1920
    factor_y = 224/1080
    input_wallcheck = cv2.resize(masked_color_image,None, fx = factor_x, fy =factor_y)
    #cv2.imshow('CNN input Wallcheck', input_wallcheck)
    img_array = image.img_to_array(input_wallcheck)
   
    img_array = np.expand_dims(img_array, axis=0)  # Create batch axis
    img_array /= 255.0  # Normalize

    print('t1',time.time()-t)
    ###for batch prediction
    imgs = np.vstack([img_array,img_array])
    with tf.device('/GPU:0'): 
     
     prediction = wall_model.predict(img_array)
    print('t2',time.time()-t)

    print(prediction)
    height, width = color_image.shape[:2]
    h = np.int0(height/2)
    w = np.int0(width/2)  
    
    print(f'prediction {prediction[0]}')
   
    if prediction[0] > 0.5:
        cv2.putText(color_image,f'Wall Check: Passed', (w-60,h+60), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 0), 3)
        cv2.putText(color_image,f'Confidence: {prediction[0]}', (w-60,h+100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 0), 3)
    else:
        cv2.putText(color_image,f'Wall Check: Failed', (w-60,h+60), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 200), 3)
        cv2.putText(color_image,f'Confidence: {prediction[0]}', (w-60,h+100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 200), 3)
    
#Initialize
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, framerate = 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, framerate = 30)

pipeline.start(config)

frame_queue = queue.Queue(maxsize=1) 
#create align object
align_to = rs.stream.color
align = rs.align(align_to)


capture_thread = threading.Thread(target=capture_frames, args = (pipeline,frame_queue,align), daemon=True)

capture_thread.start()

try:
    while True:
            if not frame_queue.empty():
        
                start = time.time()
                color_image, depth_image = frame_queue.get()
     
                #masked_color_image, hull,box,box_detected = cut_region_v2(depth_image,color_image,min_depth = 0,max_depth = 0.8)
                masked_color_image,cropped_image, hull,box,box_detected = cut_region_between_hulls(depth_image,color_image,min_depth = 0,max_depth = 0.8,  cut_rect= True)
              
                ####Add Hole Detection
                if box_detected == True:

                    #AddWallCheck
         
                    detect_walls(color_image,masked_color_image,wall_model,1)
        
                    ##
                    hole_detection = True
                    if hole_detection == True:
                        corner_1, corner_2, corner_3, corner_4 = get_corner_points(color_image, box, hull)
                    
                        edge_scale_factor = 0.2
        
                        p_new_1, p_new_2, p_new_3, p_new_4, d1,d2, x1,y1,x2,y2,x3,y3,x4,y4 = get_shifted_points(edge_scale_factor,corner_1,corner_2,corner_3, corner_4)

                        ########## draw rectangle
            
                        rect_1 = draw_rotated_rectangle(color_image,x1,x2,y1,y2,p_new_1)
                       # rect_2 = draw_rotated_rectangle(color_image,x1,x2,y1,y2,p_new_2)
                        #rect_3 = draw_rotated_rectangle(color_image,x3,x4,y3,y4,p_new_3)
                        #rect_4 = draw_rotated_rectangle(color_image,x3,x4,y3,y4,p_new_4)
                            
                        
                        image_count = detect_holes(rect_1,color_image,hole_model, img_shape,hole_threshhold = ht)
                        #image_count = detect_holes(rect_2,color_image, hole_model,img_shape,hole_threshhold = ht)
                        #image_count = detect_holes(rect_3,color_image,hole_model,img_shape,hole_threshhold = ht)
                        #image_count = detect_holes(rect_4,color_image, hole_model,img_shape,hole_threshhold = ht)
                    
             
                 
                scale_factor = 0.6
                resized_image = cv2.resize(color_image,None,fx =  scale_factor,fy = scale_factor)

                # Display the images
   
                cv2.imshow('Color Image', resized_image)
                print('iteration time: ', time.time() - start)
 
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
      
finally:
    # Stop streaming
    pipeline.stop()
    
 
    cv2.destroyAllWindows()
    