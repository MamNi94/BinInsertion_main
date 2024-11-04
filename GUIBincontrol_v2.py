import numpy as np
import pyrealsense2 as rs
import cv2

from video_capture import capture_frames

import queue
import time
import threading
import math


detection = True
if detection == True:
    from tensorflow.keras.preprocessing import image
    import tensorflow as tf
    wall_model = tf.keras.models.load_model('models\walls\inception_wall_rect_224x224_v0_L2_val_accuracy_0.993_combined_data.h5')


def detect_walls(color_image,masked_color_image, wall_model, number =1):
    t = time.time()
    wallcheck = False
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
        wallcheck = True
    else:
        cv2.putText(color_image,f'Wall Check: Failed', (w-60,h+60), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 200), 3)
        cv2.putText(color_image,f'Confidence: {prediction[0]}', (w-60,h+100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 200), 3)



    return wallcheck


def shrink_contour(points, factor):
    # Step 1: Find the centroid (center of mass)
    centroid = np.mean(points, axis=0)

    # Step 2: Move each point toward the centroid by the factor
    new_points = (1 - factor) * centroid + factor * points

    return new_points.astype(np.int32)

def shrink_contour_stable(points, factor, min_move_ratio=0.1, max_move_ratio=0.5):
    # Step 1: Find the centroid (center of mass)
    centroid = np.mean(points, axis=0)
    
    # Step 2: Calculate the distances from each point to the centroid
    distances = np.linalg.norm(points - centroid, axis=1)

    # Step 3: Calculate move distances
    move_distances = factor * distances  # Distance to move each point

    # Step 4: Introduce a min and max move ratio to control movements
    min_allowed_move = min_move_ratio * np.mean(distances)  # Limit how little points can move
    max_allowed_move = max_move_ratio * np.mean(distances)  # Limit how far points can move
    move_distances = np.clip(move_distances, min_allowed_move, max_allowed_move)  # Clamp movements

    # Step 5: Calculate the new positions
    move_directions = centroid - points  # Direction vectors toward the centroid
    move_vectors = move_directions / np.linalg.norm(move_directions, axis=1, keepdims=True)  # Normalize

    new_points = points + move_vectors * move_distances[:, np.newaxis]  # Move points inward

    return new_points.astype(np.int32)


def enlarge_contour(points, factor, max_move_ratio=2):
    # Step 1: Find the centroid (center of mass)
    centroid = np.mean(points, axis=0)
    
    # Step 2: Calculate the distance of each point from the centroid
    distances = np.linalg.norm(points - centroid, axis=1)
    
    # Step 3: Move each point outward proportionally to its distance from the centroid
    move_directions = points - centroid  # Direction vectors from the centroid to the points
    move_distances = factor * distances  # Distance to move each point
    
    # Step 4: Introduce a max move ratio to prevent excessive movement
    max_allowed_move = max_move_ratio * np.mean(distances)  # Limit how far points can move
    move_distances = np.clip(move_distances, 0, max_allowed_move)  # Clamp movements
    
    # Step 5: Calculate the new positions
    move_vectors = move_directions / distances[:, np.newaxis]  # Normalize direction vectors
    new_points = points + move_vectors * move_distances[:, np.newaxis]  # Move points outward

    return new_points.astype(np.int32)






def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def cut_region_between_hulls(depth_image, color_image, min_depth=0, max_depth=0.8, erosion_size_input=10, cut_rect = False, improved_bounding_box = False):
    masked_color_image, cropped_image, hull, box = 0, 0, 0, 0
    box_detected = False
    min_depth_mm = min_depth * 1000
    max_depth_mm = max_depth * 1000

    # Step 1: Create a binary mask based on the depth range
    mask = np.logical_and(depth_image > min_depth_mm, depth_image < max_depth_mm)

    # Step 2: Convert the mask to uint8 for further processing
    mask = mask.astype(np.uint8) * 255

    # Step 4: Find contours in the cleaned binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 6: Check if any contours were found
    if contours:
        # Optionally filter small contours (this step removes noise)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]  # Adjust the area threshold

        if contours:  # Check again after filtering
            # Step 7: Concatenate all points and create a convex hull

            largest_contour = max(contours, key=cv2.contourArea)
            all_points = largest_contour
            hull = cv2.convexHull(largest_contour)
            extraction_shape = hull
            
            if cut_rect == True:
                rect = cv2.minAreaRect(all_points)
                box = cv2.boxPoints(rect)  # Get the four vertices of the rectangle
                box = np.int0(box) 
                extraction_shape = box
                
                #get rect midpoint
                rect_center = rect[0]
                x_center = rect_center[0]
                y_center = rect_center[1]
                
     
          
            # Get the two points with the smallest y-coordinates
            
            #end experiment

            factor = 0.02  # 80% shrink (inward move)
            extraction_shape = np.array(extraction_shape)
            shape = extraction_shape
            #extraction_shape = enlarge_contour(extraction_shape, factor)

            # Step 1: Shrink the contour points
            factor = 0.12# 80% shrink (inward move)
            shrunk_contour = shrink_contour_stable(shape, min_move_ratio=0.05, factor=factor)
   
            # Step 2: Create a mask and draw the shrunk contour
            hull_mask = np.zeros_like(mask)
            cv2.drawContours(hull_mask, [extraction_shape], -1, 255, thickness=cv2.FILLED)
            shrunk_mask = np.zeros_like(mask)  # Empty mask for the shrunk contour
            cv2.drawContours(shrunk_mask, [shrunk_contour], -1, 255, thickness=cv2.FILLED)  # Shrunk filled shape
        

            # Step 9: Compute the mask for the region between the original and shrunken hulls
            region_mask = cv2.bitwise_and(hull_mask, cv2.bitwise_not(shrunk_mask))

            # Step 10: Apply the region mask to the color image
      
            if region_mask.any():
                masked_color_image = cv2.bitwise_and(color_image, color_image, mask=region_mask)
               
            else:
                masked_color_image = None
            #cv2.imshow('masked_color_image', masked_color_image)
            # Optionally crop the region between the two hulls from the color image
            # Find bounding rects of original and shrunken hulls
          
            box_detected = True
            
            #check midpoint
            height, width = color_image.shape[:2]
            if cut_rect == True:
                if abs(y_center-(height/2)) > 50 or abs(x_center-(width/2)) > 70 :
                    box_detected = False
        
    if box_detected == False:
        height, width = color_image.shape[:2]
        h = np.int0(height/2)
        w = np.int0(width/2)      
        cv2.circle(color_image, (w,h), 7, (0, 0, 200), 5)
        cv2.putText(color_image,f'No Bin Detected', (w-60,h-30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 200), 3)
        
        cv2.circle(masked_color_image, (w,h), 7, (0, 0, 200), 5)
        cv2.putText(masked_color_image,f'No Bin Detected', (w-60,h-30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 200), 3)
        
    if box_detected == True:
        height, width = color_image.shape[:2]
        h = np.int0(height/2)
        w = np.int0(width/2)      
        cv2.circle(color_image, (w,h), 7, (0, 200, 0), 5)
        cv2.putText(color_image,f'Bin Detected', (w-60,h-30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 0), 3)
        
        cv2.circle(masked_color_image, (w,h), 7, (0, 200, 0), 5)
        cv2.putText(masked_color_image,f'Bin Detected', (w-60,h-30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 0), 3)


     

    return masked_color_image, cropped_image, hull, box, box_detected



def main(inserted_bins:queue.Queue,stop_flag:threading.Event()):
    #Initialize
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, framerate = 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, framerate = 30)

    pipeline.start(config)

    #create align object
    align_to = rs.stream.color
    align = rs.align(align_to)


    # Create filters
    # 1. Temporal Filter
    temporal_filter = rs.temporal_filter()

    # 2. Spatial Filter
    spatial_filter = rs.spatial_filter()

    # 3. Hole Filling Filter
    hole_filling_filter = rs.hole_filling_filter()


    #capture_thread = threading.Thread(target=capture_frames, args = (pipeline,frame_queue,align), daemon=True)

    #capture_thread.start()
    cutting_depth = 0.8
    #get first last frame

    inserted_bins_count = 0
    box_detected_last_iteration = False
    wall_check = False

    try:
        while True:
                

                loop_start = time.time()
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                
                    continue
                
                filtered_depth_frame = temporal_filter.process(depth_frame) 
                filtered_depth_frame = spatial_filter.process(filtered_depth_frame)
                filtered_depth_frame = hole_filling_filter.process(filtered_depth_frame)

                
                raw_color_image = np.asanyarray(color_frame.get_data())
                color_image = raw_color_image
                
                depth_image = np.asanyarray(filtered_depth_frame.get_data())
                

                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.25), cv2.COLORMAP_VIRIDIS)

                scale_factor = 0.5
                resized_depth_image = cv2.resize(depth_colormap,None,fx =  scale_factor,fy = scale_factor)
                cv2.imshow('Depth Image_', resized_depth_image)
                #masked_color_image, hull,box,box_detected = cut_region_v2(depth_image,color_image,min_depth = 0,max_depth = 0.8)
                masked_color_image,cropped_image, hull,box,box_detected = cut_region_between_hulls(depth_image,color_image,min_depth = 0,max_depth = cutting_depth, erosion_size_input= 10, cut_rect= True, improved_bounding_box= False)
    
                #box_detected = False
                ####Add Hole Detection  
                print('inserted bins count:',inserted_bins_count)
              
                if box_detected == False and box_detected_last_iteration == True and wall_check == True:
                        inserted_bins_count +=1
                       # if inserted_bins.full():
                          #  inserted_bins.get()
                        inserted_bins.put(inserted_bins_count)
                        print('check 2')
                
                if box_detected == True:
                    wall_check = detect_walls(color_image,masked_color_image,wall_model,1)
                    #color_image = masked_color_image           
                    
                box_detected_last_iteration = box_detected

                scale_factor = 0.4
                resized_color_image = cv2.resize(color_image,None,fx =  scale_factor,fy = scale_factor)
                
                #print(masked_color_image)
                if masked_color_image is not None and   isinstance(masked_color_image, np.ndarray):
                
                    resized_masked_image = cv2.resize(masked_color_image,None,fx =  scale_factor,fy = scale_factor)
                    # Create the top horizontal stack
                    top_row = np.hstack((resized_masked_image, resized_color_image))
                    cv2.imshow('Color Images', top_row)
                else:
                    cv2.imshow('Color Images', resized_color_image)
                #resized_depth_image = cv2.resize(depth_colormap,None,fx =  scale_factor,fy = scale_factor)

                

                
        
                if cv2.waitKey(1) & 0xFF == ord('q') or stop_flag.is_set():
                    break
                
                    # Display the images
        
                print(time.time()-loop_start)
                #cv2.imshow('masked Image',masked_color_image)
                #cv2.imshow('Depth Image', resized_depth_image)
                #cv2.imshow('Color Image', resized_color_image)
                
                

                    
        
    except RuntimeError as e:
        print(f"Capture stopped: {e}")

    finally:
        # Stop streaming
        #pipeline.stop()
        
        print('Bincontroll stopped')

        cv2.destroyAllWindows()
    
        

        
if __name__ == "__main__":
    #stop_flag = threading.Event()
    #inserted_bins = queue.Queue(maxsize = 1)
    main()
        
