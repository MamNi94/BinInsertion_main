import numpy as np
import pyrealsense2 as rs
import cv2

import queue
import time

import math
from roi_functions import get_corner_points, get_shifted_points, draw_rotated_rectangle

detection = True
if detection == True:
    from tensorflow.keras.preprocessing import image
    import tensorflow as tf
    #wall_model = tf.keras.models.load_model('models\wall_models_2511\wall_model_inception_epoch-17_val-acc-0.9954.h5')
    wall_model = tf.keras.models.load_model('models\wall_models_1119\wall_model_inception_epoch-32_val-acc-0.9722_299x299.h5')

    hole_model = tf.keras.models.load_model('models\hole_models_1911\hole_model_inception_epoch-18_val-acc-0.9846.h5')
    




def detect_holes_batch(rects, img, hole_model, img_shape=224, hole_threshold=0.1):
    """
    Detect holes in multiple rectangles with a single model inference.
    
    Parameters:
        rects (list): A list of rectangular regions (4 points per rectangle).
        img (ndarray): The input image.
        hole_model (keras.Model): The trained model for hole detection.
        img_shape (int): Target shape (width/height) for model input.
        hole_threshold (float): Threshold for determining pass/fail.
    
    Returns:
        ndarray: The modified image with annotations.
    """
    # List to store cropped images
    cropped_images = []
    valid_rects = []
    hole_check = []

    for rect in rects:
        rect = np.array(rect, dtype=np.float32)
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
        
        # Resize cropped image to the desired shape
        resized_image = cv2.resize(cropped_image, (img_shape, img_shape))
        
        # Check dimensions
        if resized_image.shape[0] == img_shape and resized_image.shape[1] == img_shape:
            cropped_images.append(resized_image)
            valid_rects.append(rect)  # Only keep valid rects

    # Convert cropped images to a batch for model inference
    h = 0
    if cropped_images:
        cropped_images_array = np.array([image.img_to_array(img) / 255.0 for img in cropped_images])
        predictions = hole_model.predict(cropped_images_array)

        for rect, prediction in zip(valid_rects, predictions):
            text_offset_x = 150
            text_offset_y = 80
            if prediction[0] > hole_threshold:
                cv2.putText(img, 'Passed', (int(rect[0][0] - text_offset_x), int(rect[0][1] + text_offset_y)), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
                hole_check.append(1)
            else:
                cv2.putText(img, 'Failed', (int(rect[0][0] - text_offset_x), int(rect[0][1] + text_offset_y)), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
                hole_check.append(0)
            h+=1
    else:
        print("No valid cropped images found for inference.")

    return hole_check




def detect_walls(color_image,masked_color_image, wall_model, number =1):
    
    wall_check = None

    height, width, _ = masked_color_image.shape
    factor_x = 299/width
    factor_y = 299/height
    input_wallcheck = cv2.resize(masked_color_image,None, fx = factor_x, fy =factor_y)
    #cv2.imshow('CNN input Wallcheck', input_wallcheck)
    img_array = image.img_to_array(input_wallcheck)
   
    img_array = np.expand_dims(img_array, axis=0)  # Create batch axis
    img_array /= 255.0  # Normalize

    ###for batch prediction
    with tf.device('/GPU:0'): 
      prediction = wall_model.predict(img_array)


    print(prediction)
    height, width = color_image.shape[:2]
    h = np.int0(height/2)
    w = np.int0(width/2)  
    
   
    if prediction[0] > 0.999:
        cv2.putText(color_image,f'Wall Check: Passed', (w-60,h+60), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 0), 3)
        cv2.putText(color_image,f'Confidence: {prediction[0]}', (w-60,h+100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 0), 3)
        wall_check = 1
    else:
        cv2.putText(color_image,f'Wall Check: Failed', (w-60,h+60), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 200), 3)
        cv2.putText(color_image,f'Confidence: {prediction[0]}', (w-60,h+100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 200), 3)
        wall_check = 0

    return wall_check

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



def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)



def scale_hull(hull,scale_factor = 1, adjustent_factor = 1.01):
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

def detect_box(depth_image, color_image, min_depth=0, max_depth=0.8):
    hull, box = 0, 0
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

            rect = cv2.minAreaRect(all_points)
            box = cv2.boxPoints(rect)  # Get the four vertices of the rectangle
            box = np.intp(box) 
        
            #get rect midpoint
            rect_center = rect[0]
            x_center = rect_center[0]
            y_center = rect_center[1]
                



            box_detected = True
            
            #check midpoint
            height, width = color_image.shape[:2]
            
            if abs(y_center-(height/2)) > 50 or abs(x_center-(width/2)) > 70 :
                box_detected = False
        
    if box_detected == False:
        height, width = color_image.shape[:2]
        h = np.intp(height/2)
        w = np.intp(width/2)      
        cv2.circle(color_image, (w,h), 7, (0, 0, 200), 5)
        cv2.putText(color_image,f'No Bin Detected', (w-60,h-30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 200), 3)

    if box_detected == True:
        height, width = color_image.shape[:2]
        h = np.intp(height/2)
        w = np.intp(width/2)      
        cv2.circle(color_image, (w,h), 7, (0, 200, 0), 5)
        cv2.putText(color_image,f'Bin Detected', (w-60,h-30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 0), 3)
    
    return hull, box, box_detected

def get_rect_center(rect):
    """
    Calculate the center of a rectangle given its 4 points.
    
    Parameters:
        rect (list): A list of 4 points defining the rectangle (in any consistent order).
        
    Returns:
        tuple: (cx, cy) the coordinates of the rectangle's center.
    """
    # Ensure the points are numpy arrays for easy manipulation
    rect = np.array(rect, dtype=np.float32)
    
    # Calculate the center as the average of the top-left and bottom-right corners
    top_left = rect[0]
    bottom_right = rect[2]
    cx = (top_left[0] + bottom_right[0]) / 2
    cy = (top_left[1] + bottom_right[1]) / 2
    
    return cx, cy


 # Function to draw the progress bar on the image
def draw_progress_bar(image, percentage,bin_check_final):

    #progress bar
    # Progress bar parameters
    bar_width = 400
    bar_height = 30
    bar_x = 800  # X position of the bar
    bar_y = 350  # Y position of the bar
    bar_color = (0, 255, 0)
    if bin_check_final == False:
        bar_color = (0, 0, 255)

    background_color = (50, 50, 50)
    text_color = (255, 255, 255)
  
    # Make a copy of the image to avoid overwriting
    display_image = image
    
    # Draw the background of the progress bar
    cv2.rectangle(display_image, (bar_x, bar_y), 
                (bar_x + bar_width, bar_y + bar_height), background_color, -1)
    
    # Calculate the width of the progress portion
    progress_width = int((percentage / 10) * bar_width)
    
    # Draw the progress
    cv2.rectangle(display_image, (bar_x, bar_y), 
    (bar_x + progress_width, bar_y + bar_height), bar_color, -1)

    # Add text to show the percentage
    text = f"{percentage*10:.1f}%"
    font_scale = 0.8
    thickness = 2
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = bar_x + (bar_width - text_size[0]) // 2
    text_y = bar_y + bar_height // 2 + text_size[1] // 2
    cv2.putText(display_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

def is_hull_bounded_by_rect(image, cut_region_final, rect_width=200, rect_height=150):
    """
    Check if the convex hull is fully within a rectangle around the midpoint of the image.

    Parameters:
    - image (numpy.ndarray): The input image.
    - cut_region_final (numpy.ndarray): The points of the convex hull (array of shape (N, 1, 2)).
    - rect_width (int): The width of the rectangle around the midpoint (default 200).
    - rect_height (int): The height of the rectangle around the midpoint (default 150).

    Returns:
    - bool: True if the convex hull is fully within the rectangle, False otherwise.
    """
    # Get image dimensions
    image_height, image_width = image.shape[:2]

    # Calculate the midpoint of the image
    mid_x, mid_y = image_width // 2, image_height // 2

    # Define the rectangle around the midpoint
    rect_top_left = (mid_x - rect_width // 2, mid_y - rect_height // 2)
    rect_bottom_right = (mid_x + rect_width // 2, mid_y + rect_height // 2)

    # Get the bounding rectangle of the convex hull
    
    x, y, w, h = cv2.boundingRect(np.array(cut_region_final))
    hull_rect_top_left = (x, y)
    hull_rect_bottom_right = (x + w, y + h)

    # Check if the hull is bounded by the rectangle
    is_bounded = (
        hull_rect_top_left[0] >= rect_top_left[0]
        and hull_rect_top_left[1] >= rect_top_left[1]
        and hull_rect_bottom_right[0] <= rect_bottom_right[0]
        and hull_rect_bottom_right[1] <= rect_bottom_right[1]
    )

    cv2.rectangle(image, rect_top_left, rect_bottom_right, (255, 0, 0), 2)  # Midpoint rectangle
    cv2.polylines(image, [cut_region_final], isClosed=True, color=(0, 255, 0), thickness=2)  # Convex hull
    


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

previous_center = [0,0]

classification_matrix = []

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
            hull,box,box_detected = detect_box(depth_image,color_image,min_depth = 0,max_depth = cutting_depth)

            if box_detected == True:
                current_center =  get_rect_center(box)
                center_dist = distance(current_center,previous_center)
                previous_center = current_center
                print('center_dist:', center_dist)
            
            else:
                classification_matrix = []
            #box_detected = False
            ####Add Hole Detection
            cropped_cut_region_final = None
          
            detection = True
            hole_detection = True
            wall_detection = True

            
            
            if box_detected == True and center_dist <20:
                
               
                if detection == True:
                    corners = get_corner_points(color_image, box, hull)
                    
                    ###cut bin
                    hull = cv2.convexHull(np.array(corners))

                    # Find the centroid of the hull
                    centroid = np.mean(hull[:, 0, :], axis=0)

                    
                    scale_factor = 1.015
                    adjustment_factor = 1.02
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
                    cut_region_final = cv2.bitwise_and(color_image, color_image, mask=mask_between)
                    
                    #is_hull_bounded_by_rect(color_image, expanded_hull, rect_width=1300, rect_height=880)

                    
                    
                    #Cut Bin end

                    edge_scale_factor = 0.19
               
                    p_new_1, p_new_2, p_new_3, p_new_4, d1,d2, x1,y1,x2,y2,x3,y3,x4,y4 = get_shifted_points(edge_scale_factor,corners)

                    ########## draw rectangle
        
                    rect_1 = draw_rotated_rectangle(color_image,x1,x2,y1,y2,p_new_1)
                    rect_2 = draw_rotated_rectangle(color_image,x1,x2,y1,y2,p_new_2)
                    rect_3 = draw_rotated_rectangle(color_image,x3,x4,y3,y4,p_new_3)
                    rect_4 = draw_rotated_rectangle(color_image,x3,x4,y3,y4,p_new_4)

                    classification_list = []
                        
                    if hole_detection == True:
          
                        hole_result = detect_holes_batch([rect_1,rect_2,rect_3,rect_4],color_image,hole_model)
                        if hole_result is not None:
                            classification_list = hole_result
                        
                       
                    if wall_detection == True:
                        
                        height, width, _ = cut_region_final.shape 

                        y_start= 0
                        x_start= width // 2 - height//2 -200

                        y_end = height
                        x_end = width // 2 + height//2 + 200

                        cropped_cut_region_final = cut_region_final[y_start:y_end, x_start:x_end]
                        


                        wall_result = detect_walls(color_image,cropped_cut_region_final,wall_model,1)
                        if wall_result is not None:
                            classification_list.append(wall_result)

                    

                    classification_matrix.append(classification_list)
                    bin_check_final = None
                    if len(classification_matrix) > 10:
                        
                        column_sums = [sum(column) for column in zip(*classification_matrix)]
                        for index, col_sum in enumerate(column_sums):
                            if col_sum >= 5:
                                bin_check_final = True
                                
                            else:
                                
                                cv2.putText(color_image,f'This Bin Is Faulty!', (850,420), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 3)
                                bin_check_final = False
                                break
                        
                        if bin_check_final == True:
                             cv2.putText(color_image,f'This Bin Is Amazing!', (850,420), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 3)

                        classification_matrix.pop(0)

                    
                    draw_progress_bar(color_image, len(classification_matrix),bin_check_final)

            ##add legend
            cv2.putText(color_image,f'good wall:', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 3)
            cv2.putText(color_image,f'good hole:', (100,150), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 3)
            
            cv2.circle(color_image, (300,90), 1, (0, 255, 0), 30)
            cv2.circle(color_image, (300,140), 1, (255, 125, 0), 15)
            cv2.circle(color_image, (340,140), 1, (0, 165, 255), 15)
            #end legend

              
        
            scale_factor = 0.8
            resized_color_image = cv2.resize(color_image,None,fx =  scale_factor,fy = scale_factor)
           

            if cropped_cut_region_final is not None and   isinstance(cropped_cut_region_final, np.ndarray):
                scale_factor = 0.4
                scaled_result = cv2.resize(cropped_cut_region_final,None, fx = scale_factor, fy = scale_factor)
                # Create the top horizontal stack
                top_row = np.hstack(( resized_depth_image,scaled_result))

                #cropp_row = np.hstack((cropp_1,cropp_2,cropp_3,cropp_4))
                #resized_color_image = scaled_result
               
                #final_display = np.vstack((top_row, bottom_row))
                cv2.imshow('Depth_crop', top_row)
                #cv2.imshow('Color Images__', cropp_row)
                #cv2.imshow('Color image',scaled_result)

                
            cv2.imshow('Color Image', resized_color_image) 
          
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
    
        
        
