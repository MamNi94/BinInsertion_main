import numpy as np
import pyrealsense2 as rs
import cv2
import os
import heapq
import matplotlib as plt
from tensorflow.keras.preprocessing import image
import tensorflow as tf

from video_capture import capture_frames
from roi_functions import cut_region_at_distance, detect_roi,cut_region_v2, get_shifted_points,get_corner_points, draw_rotated_rectangle, cut_region_between_hulls
import threading
import queue
import time

ht = 0.5


#wall_model = tf.keras.models.load_model('VGG_bincheck_wall_224x224_v1.h5')
#wall_model_1 = tf.keras.models.load_model('models/wall_models/models_cut/inception_wall_224x224_cut_out_v3.h5')
#wall_model_2 = tf.keras.models.load_model('models/wall_models/models_cut/inception_wall_224x224_cut_v6_L2_val_accuracy_0.9799723625183105.h5')
#wall_model_3 = tf.keras.models.load_model('models/wall_models/models_cut/inception_wall_rect_12_224x224_v0_L2_val_accuracy_0.9943740963935852.h5')
wall_model = tf.keras.models.load_model('models/wall_models/models_cut/inception_wall_rect_224x224_v0_L2_val_accuracy_0.993_combined_data.h5')
#wall_model_3 = tf.keras.models.load_model('models/wall_models/models_cut/dense.h5')
#wall_model_3 = tf.keras.models.load_model('models/wall_models/models_whole_bin_thresh/inception_wall_224x224_thresh_whole_bin_v1_L2_val_accuracy_0.981792688369751 (1).h5')
#wall_model = tf.keras.models.load_model('models/wall_models/models_cut/inception_walls_224x224_cut_out_v2.h5')

img_shape = 224
if img_shape == 224:
    pass
    #hole_model = tf.keras.models.load_model('VGG_bincheck_holes_224x224_v2.h5')
    #hole_model = tf.keras.models.load_model('models_0909/Inception_bincheck_holes_224x224_256_v2.h5')
    #hole_model = tf.keras.models.load_model('models_0909/inception_Net_bincheck_holes_224x224_v0.h5')
    #hole_model = tf.keras.models.load_model('models_0909/VGG16_bincheck_holes_224x224_1024_v0.h5') #good
    #hole_model = tf.keras.models.load_model('models_0909/inception_bincheck_holes_224x224_v1.h5')
    #hole_model = tf.keras.models.load_model('models_0909/VGG19_bincheck_holes_224x224_v0.h5') #Best performance
    #hole_model = tf.keras.models.load_model('models/hole_models/inception_hole_224x224_512_v0.h5') 
    #hole_model = tf.keras.models.load_model('models/hole_models/inception_hole_224x224_1024_v0.h5') 
    #hole_model = tf.keras.models.load_model('models/hole_models/inception_hole_224x224_close.h5') 
if img_shape == 112:
    #hole_model = tf.keras.models.load_model('VGG_bincheck_holes_112x112_v0.h5')
    pass

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
    print(cropped_image.shape)
    
    cropped_image = cv2.resize(cropped_image,None,fx = img_shape/cropped_image.shape[1],fy = img_shape/cropped_image.shape[0])
    print(cropped_image.shape)
    inspect_holes = True
    if cropped_image.shape[0] !=img_shape or cropped_image.shape[1]  != img_shape:
        inspect_holes = False
    #model
    if inspect_holes == True:
        img_array = image.img_to_array(cropped_image)
        
        img_array = np.expand_dims(img_array, axis=0)  # Create batch axis
        img_array /= 255.0  # Normalize
        prediction = hole_model.predict(img_array)
      
        text_offset_x = 150
        text_offset_y = 80
        if prediction[0] > hole_threshhold:
            cv2.putText(img,f'Passed', (int(rect[0][0]-text_offset_x),int(rect[0][1] + text_offset_y)), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 0), 2)
        else:
            cv2.putText(img,f'Failed', (int(rect[0][0]-text_offset_x),int(rect[0][1] + text_offset_y)), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 200), 2)
  
def detect_walls(color_image,masked_color_image, wall_model, number =1):
    factor_x = 224/1920
    factor_y = 224/1080
    input_wallcheck = cv2.resize(masked_color_image,None, fx = factor_x, fy =factor_y)
    #cv2.imshow('CNN input Wallcheck', input_wallcheck)
    img_array = image.img_to_array(input_wallcheck)
   
    img_array = np.expand_dims(img_array, axis=0)  # Create batch axis
    img_array /= 255.0  # Normalize
    prediction = wall_model.predict(img_array)
    
    
    height, width = color_image.shape[:2]
    h = np.int(height/2)
    w = np.int(width/2)  
    
    print(f'prediction {prediction[0]}')
    if prediction[0] > 0.9:
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
                masked_color_image,cropped_image, hull,box,box_detected = cut_region_between_hulls(depth_image,color_image,min_depth = 0,max_depth = 0.8, erosion_size= 12, cut_rect= True)
                
                
                #color_image = masked_color_image
                ####Add Hole Detection
                if box_detected == True:
                    
                    
                    #AddWallCheck
                    #detect_walls(masked_color_image,wall_model_3,1)
                    #detect_walls(masked_color_image,wall_model_2,2)
                    #masked_color_image_cut,_, _,_,_ = cut_region_between_hulls(depth_image,color_image,min_depth = 0,max_depth = 0.7, erosion_size= 12,cut_rect = True)
                    detect_walls(color_image,masked_color_image,wall_model,1)
        
                    ###whole bin / Tresh
                    '''
                    masked_color_image_bin,cropped_image_bin, hull_bin,box_bin,box_detected = cut_region_between_hulls(depth_image,color_image,min_depth = 0,max_depth = 0.7, erosion_size= 1000, cut_rect = True)
                    masked_gray =  cv2.cvtColor(masked_color_image_bin, cv2.COLOR_BGR2GRAY)
                    retval,tresh = cv2.threshold(masked_gray, 125, 255, cv2.THRESH_BINARY)
                    cv2.imshow('Thresh', tresh)
                    detect_walls(masked_color_image,wall_model_3,3)
                    
                    '''
                    #color_image = masked_color_image
                    ##
                    hole_detection = False
                    if hole_detection == True:
                        corner_1, corner_2, corner_3, corner_4 = get_corner_points(color_image, box, hull)
                    
                        edge_scale_factor = 0.2
        
                        p_new_1, p_new_2, p_new_3, p_new_4, d1,d2, x1,y1,x2,y2,x3,y3,x4,y4 = get_shifted_points(edge_scale_factor,corner_1,corner_2,corner_3, corner_4)

                        ########## draw rectangle
            
                        rect_1 = draw_rotated_rectangle(color_image,x1,x2,y1,y2,p_new_1)
                        rect_2 = draw_rotated_rectangle(color_image,x1,x2,y1,y2,p_new_2)
                        rect_3 = draw_rotated_rectangle(color_image,x3,x4,y3,y4,p_new_3)
                        rect_4 = draw_rotated_rectangle(color_image,x3,x4,y3,y4,p_new_4)
                            
                        
                        #image_count = detect_holes(rect_1,color_image,hole_model, img_shape,hole_threshhold = ht)
                        #image_count = detect_holes(rect_2,color_image, hole_model,img_shape,hole_threshhold = ht)
                        #image_count = detect_holes(rect_3,color_image,hole_model,img_shape,hole_threshhold = ht)
                        #image_count = detect_holes(rect_4,color_image, hole_model,img_shape,hole_threshhold = ht)
                    print('end time normal', time.time() - start)
             
                 
                scale_factor = 0.6
                resized_image = cv2.resize(color_image,None,fx =  scale_factor,fy = scale_factor)

                # Display the images
   
                cv2.imshow('Color Image', resized_image)
                #cv2.imshow('masked Image',masked_color_image)
 
           
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
      
finally:
    # Stop streaming
    pipeline.stop()
    

 
    cv2.destroyAllWindows()
    