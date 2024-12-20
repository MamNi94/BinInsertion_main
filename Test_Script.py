import numpy as np
import pyrealsense2 as rs
import cv2

from video_capture import capture_frames

import queue
import time
import threading
import math
from roi_functions import get_corner_points, get_shifted_points, draw_rotated_rectangle

detection = False
if detection == True:
    from tensorflow.keras.preprocessing import image
    import tensorflow as tf
    wall_model = tf.keras.models.load_model('models\wall_models_1111\inception_wall_rect_224x224_v0_L2_val_accuracy_0.9932_1111_combined.h5')

    hole_model = tf.keras.models.load_model('models\hole_models_1111\inception_wall_rect_224x224_v0_L2_val_accuracy_0.9897_1111.h5')


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

def calculate_angle(p1, p2, p3):
    # Vector between p1->p2 and p2->p3
    v1 = p1 - p2
    v2 = p3 - p2

    # Compute the dot product and magnitudes of the vectors
    dot_product = np.dot(v1, v2)
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)

    # Calculate the angle in radians and convert to degrees
    angle = np.arccos(dot_product / (mag_v1 * mag_v2))
    angle_deg = np.degrees(angle)

    return angle_deg

def get_max_points(box):
    max_x_index = np.argmax(box[:, 0])  # Select the first column (x-coordinates)
    # Get the point with the maximum x-coordinate
    max_x_point = box[max_x_index]

    # Step 1: Extract x-coordinates
    x_coordinates = box[:, 0]

    unique_x = np.sort(x_coordinates)[::-1] 

    # Step 3: Check if there are at least two unique x-coordinates
    if len(unique_x) >= 2:
        # Step 4: Get the second largest x-coordinate
        second_largest_x = unique_x[1]

        # Step 5: Retrieve the corresponding point(s) with that x-coordinate
        second_largest_point = box[box[:, 0] == second_largest_x][0]
        if max_x_point[1] == second_largest_point[1]:
            second_largest_point = box[box[:, 0] == second_largest_x][1]
        
    return max_x_point, second_largest_point

def get_max_points_(box):
    min_y_index = np.argmin(box[:, 1])  # Select the first column (x-coordinates)
    # Get the point with the maximum x-coordinate
    max_y_point = box[min_y_index]

    # Step 1: Extract x-coordinates
    y_coordinates = box[:, 1]

    unique_y = np.sort(y_coordinates)[::-1] 

    # Step 3: Check if there are at least two unique x-coordinates
    if len(unique_y) >= 2:
        # Step 4: Get the second largest x-coordinate
        second_largest_y = unique_y[2]

        # Step 5: Retrieve the corresponding point(s) with that x-coordinate
        second_largest_point = box[box[:, 1] == second_largest_y][0]
        if max_y_point[1] == second_largest_point[1]:
            second_largest_point = box[box[:, 1] == second_largest_y][1]
        
    return max_y_point, second_largest_point

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
                print(extraction_shape)
                
                ####Improved Bounding Box
                if improved_bounding_box ==True:
                    for i in range(4):
                        p = box[i]      # Current point
                    
                        #cv2.putText(color_image,f'corner: {i}', p, cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 200), 3)                

                    max_1, max_2 = get_max_points(box)
                    print(max_1, max_2)
                
                    bin_side_length = np.round(distance(max_1,max_2))
                        
                    #cv2.putText(color_image,f'p1', (max_1[0], max_1[1]+30) , cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 0), 3)
                    #cv2.putText(color_image,f'p2', (max_2[0],max_2[1] + 30) , cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 0), 3)
                    #cv2.putText(color_image,f'right wall length: {bin_side_length}', (np.int0(max_2[0])-100,np.int0((max_2[1]+max_1[1])/2) ) , cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 0), 3)
                    #cv2.putText(color_image,f'top wall length: {bin_side_length}', (np.int0((max_2[0]+max_1[0])/2), (np.int0(max_2[1])+100)), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 0), 3)
                    
                    if max_1[1] > max_2[1]:
                        direction = np.array([-(max_1[1]-max_2[1])/bin_side_length,(max_1[0] - max_2[0])/bin_side_length])
                    else:
                        direction = np.array([(max_1[1]-max_2[1])/bin_side_length,-(max_1[0] - max_2[0])/bin_side_length])
                    #if max_1[0] > max_2[0]:
                    #  direction = np.array([-(max_1[1]-max_2[1])/bin_side_length,(max_1[0] - max_2[0])/bin_side_length])
                        #print('heey')
                    #else:
                        #direction = np.array([(((max_1[1]-max_2[1])/bin_side_length)-0.01),-(max_1[0] - max_2[0])/bin_side_length])
                    # print('hoo')
                   
                    bin_factor = 1.45 #1.45
                    #bin_factor = 1/1.45
                    calculated_point_1 = (direction * bin_factor * bin_side_length)  + max_1
                    calculated_point_2 = (direction * bin_factor * bin_side_length)  + max_2
                    
                    #cv2.circle(color_image, (np.int0(calculated_point_1[0]),np.int0(calculated_point_1[1])), 3,(255,255,0),3)
                    #cv2.circle(color_image, (np.int0(calculated_point_2[0]),np.int0(calculated_point_2[1])), 3,(255,0,255),3)
                    
                    print(calculated_point_1, calculated_point_2, max_1,max_2)
                    all_points = np.array([np.int0(calculated_point_1), np.int0(calculated_point_2), max_2,max_1])
                    box = cv2.convexHull(all_points)
                    #box = cv2.boxPoints(rect)  # Get the four vertices of the rectangle
                    #box = np.int0(box) 
                    if improved_bounding_box ==True:
                        extraction_shape = box
                    
                    #####Improved Bounding Box End
                
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


    
            K =11
            kernel = np.ones((K, K), np.uint8)  # You can adjust the kernel size

            # Apply dilation followed by erosion (this is called "closing")
            depth_colormap = cv2.morphologyEx(depth_colormap, cv2.MORPH_CLOSE, kernel)
            #depth_colormap = cv2.morphologyEx(depth_colormap, cv2.MORPH_CLOSE, kernel)
     
            #depth_colormap = cv2.morphologyEx(depth_colormap, cv2.MORPH_CLOSE, kernel)
            depth_image = cv2.morphologyEx(depth_image, cv2.MORPH_CLOSE, kernel)
            #depth_image = cv2.morphologyEx(depth_image, cv2.MORPH_CLOSE, kernel)
           

            hsv_map = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2HSV)

            # Define HSV range for blue
            lower_blue = np.array([40, 50, 50])    # Adjust these values if needed
            upper_blue = np.array([130, 255, 255]) # to precisely match your blue shade

            # Create a mask to isolate blue regions
            blue_mask = cv2.inRange(hsv_map, lower_blue, upper_blue)

            # Apply the mask to retain only the blue areas
            blue_only = cv2.bitwise_and(depth_colormap, depth_colormap, mask=blue_mask)
            blue_gray = cv2.cvtColor(blue_only, cv2.COLOR_BGR2GRAY)
            #depth_image = cv2.normalize(blue_gray, None, 0, 255, cv2.NORM_MINMAX)
            
            #Improve Depth Image

            scale_factor = 0.4
            resized_depth_image = cv2.resize(depth_colormap,None,fx =  scale_factor,fy = scale_factor)
           
            #masked_color_image, hull,box,box_detected = cut_region_v2(depth_image,color_image,min_depth = 0,max_depth = 0.8)
            masked_color_image,cropped_image, hull,box,box_detected = cut_region_between_hulls(depth_image,color_image,min_depth = 0,max_depth = cutting_depth, erosion_size_input= 10, cut_rect= True, improved_bounding_box= False)

           
            #box_detected = False
            ####Add Hole Detection
            
            detection = True
            if box_detected == True and detection == True:


                wall_detection = False
                if wall_detection == True:
                    detect_walls(color_image,masked_color_image,wall_model,1)
                #color_image = masked_color_image

                hole_detection = True
                if hole_detection == True:
                    corners = get_corner_points(color_image, box, hull)

                    ###Hull Tetst
                    
                    
                   
                    
                   
                    
             
                    def scale_hull(hull,scale_factor, adjustent_factor = 1.01):
                        expanded_hull = []

                        centroid = np.mean(hull[:, 0, :], axis=0)

                        points_below_centroid = hull[hull[:, 0, 1] < centroid[1]]

                        points_below_centroid_flat = points_below_centroid[:, 0, :]
                        # Get the x values
                        x_values = points_below_centroid_flat[:, 0]

                        # Get the smallest x value
                        min_x_value = np.min(x_values)

                        # Get the largest x value
                        max_x_value = np.max(x_values)

                        for point in hull[:, 0, :]:
                            vector = point - centroid  # Vector from centroid to the point
                            if point[1] <= centroid[1] and point[0] >min_x_value and point[0]< max_x_value:

                                expanded_point = centroid + scale_factor * vector  * adjustent_factor
                                #expanded_point = centroid + scale_factor * vector
                            else:
                                expanded_point = centroid + scale_factor * vector  # Scale outward
                            expanded_hull.append(expanded_point)
                        

                        expanded_hull = np.array(expanded_hull, dtype=np.int32)

                        return expanded_hull

                    hull = cv2.convexHull(np.array(corners))

                    # Find the centroid of the hull
                    centroid = np.mean(hull[:, 0, :], axis=0)


                    # Scale factor (adjust this to control how far the hull moves outward)
                    
                    scale_factor = 1.015
                    adjustment_factor = 1.01
                    # Expand the hull outward
                    expanded_hull = scale_hull(hull, scale_factor, adjustment_factor)
                    scale_factor = 0.86
                    adjustment_factor = 1.01
                    shrunk_hull = scale_hull(hull, scale_factor,adjustment_factor)

                    # Draw the convex hull on the image
                    #cv2.polylines(color_image, [expanded_hull], isClosed=True, color=(0, 0, 255), thickness=2)
                    #cv2.polylines(color_image, [shrunk_hull], isClosed=True, color=(0, 0, 255), thickness=2)

                    ##cut between hulls
                    # Create masks
                    mask_outer = np.zeros(color_image.shape[:2], dtype=np.uint8)
                    mask_inner = np.zeros(color_image.shape[:2], dtype=np.uint8)

                    # Fill the masks with the hulls
                    cv2.fillPoly(mask_outer, [expanded_hull], 255)  # Outer hull (expanded)
                    cv2.fillPoly(mask_inner, [shrunk_hull], 255)   # Inner hull (shrunk)

                    # Subtract the inner mask from the outer mask
                    mask_between = cv2.subtract(mask_outer, mask_inner)

                    # Apply the mask to the image
                    result = cv2.bitwise_and(color_image, color_image, mask=mask_between)
                    
                    
                    #Hull Test End

                    edge_scale_factor = 0.2
                    edge_scale_factor = 0.19
               
                    p_new_1, p_new_2, p_new_3, p_new_4, d1,d2, x1,y1,x2,y2,x3,y3,x4,y4 = get_shifted_points(edge_scale_factor,corners)

                    ########## draw rectangle
        
                    rect_1 = draw_rotated_rectangle(color_image,x1,x2,y1,y2,p_new_1)
                    rect_2 = draw_rotated_rectangle(color_image,x1,x2,y1,y2,p_new_2)
                    rect_3 = draw_rotated_rectangle(color_image,x3,x4,y3,y4,p_new_3)
                    rect_4 = draw_rotated_rectangle(color_image,x3,x4,y3,y4,p_new_4)
                        
                    
                    #image_count = detect_holes(rect_1,color_image,hole_model)
                    #image_count = detect_holes(rect_2,color_image, hole_model,img_shape,hole_threshhold = ht)
                    #image_count = detect_holes(rect_3,color_image,hole_model,img_shape,hole_threshhold = ht)
                    #image_count = detect_holes(rect_4,color_image, hole_model,img_shape,hole_threshhold = ht)


            
            height, width, _ = result.shape 

            y_start= 0
            x_start= width // 2 - height//2 -200

            y_end = height
            x_end = width // 2 + height//2 + 200

            cropped_result = result[y_start:y_end, x_start:x_end]
            cropped_masked = masked_color_image[y_start:y_end, x_start:x_end]
        
            scale_factor = 0.4
            resized_color_image = cv2.resize(color_image,None,fx =  scale_factor,fy = scale_factor)
           
            
            #print(masked_color_image)
            if masked_color_image is not None and   isinstance(masked_color_image, np.ndarray):
                scaled_result = cv2.resize(cropped_result,None, fx = scale_factor, fy = scale_factor)
                resized_masked_image = cv2.resize(cropped_masked,None,fx =  scale_factor,fy = scale_factor)
                # Create the top horizontal stack
                top_row = np.hstack(( resized_depth_image,resized_color_image))
                bottom_row = np.hstack((resized_masked_image, scaled_result))
                #final_display = np.vstack((top_row, bottom_row))
                cv2.imshow('Color Images', top_row)
                cv2.imshow('Color Images__', bottom_row)
            else:
                cv2.imshow('Color Images', resized_color_image)
            #resized_depth_image = cv2.resize(depth_colormap,None,fx =  scale_factor,fy = scale_factor)
  

            

            
    
            if cv2.waitKey(1) & 0xFF == ord('q') :
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
    
        
        
