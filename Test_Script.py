import numpy as np
import pyrealsense2 as rs
import cv2

from video_capture import capture_frames

import queue
import time
import threading
import math

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

    # Step 3: Optional - Clean the mask using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove small noise

    # Step 4: Find contours in the cleaned binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 6: Check if any contours were found
    if contours:
        # Optionally filter small contours (this step removes noise)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]  # Adjust the area threshold

        if contours:  # Check again after filtering
            # Step 7: Concatenate all points and create a convex hull
            all_points = np.concatenate(contours)
            hull = cv2.convexHull(all_points)
            
            hull_area = cv2.contourArea(hull)
            erosion_size = int(np.sqrt(hull_area) / erosion_size_input)
            
            extraction_shape = hull
            
            if cut_rect == True:
                rect = cv2.minAreaRect(all_points)
                box = cv2.boxPoints(rect)  # Get the four vertices of the rectangle
                box = np.int0(box) 
                extraction_shape = box
                
                #dynamic erosion size
                d1 = distance(box[0],box[1])
                d2 = distance(box[0],box[2])
                d = max(d1,d2)
                erosion_size = np.int(d/erosion_size_input)
                
                ####Improved Bounding Box

                for i in range(4):
                    p1 = box[i - 1]  # Previous point (wraps around using i-1)
                    p2 = box[i]      # Current point
                    p3 = box[(i + 1) % 4]  # Next point (wraps around using i+1)
                    
                    #angle = np.round(calculate_angle(p1, p2, p3))
                    cv2.putText(color_image,f'corner: {i}', p2, cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 200), 3)
                

                max_1, max_2 = get_max_points(box)
                print(max_1, max_2)
               
                bin_side_length = np.round(distance(max_1,max_2))
                    
                cv2.putText(color_image,f'p1', (max_1[0], max_1[1]+30) , cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 0), 3)
                cv2.putText(color_image,f'p2', (max_2[0],max_2[1] + 30) , cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 0), 3)
                cv2.putText(color_image,f'right wall length: {bin_side_length}', (np.int(max_2[0])-100,np.int((max_2[1]+max_1[1])/2) ) , cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 0), 3)

                if max_1[1] > max_2[1]:
                    direction = np.array([-(max_1[1]-max_2[1])/bin_side_length,(max_1[0] - max_2[0])/bin_side_length])
                else:
                    direction = np.array([(max_1[1]-max_2[1])/bin_side_length,-(max_1[0] - max_2[0])/bin_side_length])
                calculated_point_1 = direction * 1.45 * bin_side_length  + max_1
                calculated_point_2 = direction * 1.45 * bin_side_length  + max_2
                
                cv2.circle(color_image, (np.int(calculated_point_1[0]),np.int(calculated_point_1[1])), 3,(255,255,0),3)
                cv2.circle(color_image, (np.int(calculated_point_2[0]),np.int(calculated_point_2[1])), 3,(255,255,0),3)
                
                print(calculated_point_1, calculated_point_2, max_1,max_2)
                all_points = np.array([np.int0(calculated_point_1), np.int0(calculated_point_2), max_1,max_2])
                rect = cv2.minAreaRect(all_points)
                box = cv2.boxPoints(rect)  # Get the four vertices of the rectangle
                box = np.int0(box) 
                if improved_bounding_box ==True:
                    extraction_shape = box
                
                #####Improved Bounding Box End
                
                #get rect midpoint
                rect_center = rect[0]
                x_center = rect_center[0]
                y_center = rect_center[1]
                

            # Create a mask for the original convex hull
            hull_mask = np.zeros_like(mask)
            cv2.drawContours(hull_mask, [extraction_shape], -1, 255, thickness=cv2.FILLED)

            # Step 8: Shrink the convex hull by using erosion
            kernel = np.ones((erosion_size, erosion_size), np.uint8)  # Erosion kernel
            eroded_hull_mask = cv2.erode(hull_mask, kernel, iterations=1)

            # Step 9: Compute the mask for the region between the original and shrunken hulls
            region_mask = cv2.bitwise_and(hull_mask, cv2.bitwise_not(eroded_hull_mask))

            # Step 10: Apply the region mask to the color image
            masked_color_image = cv2.bitwise_and(color_image, color_image, mask=region_mask)

            # Optionally crop the region between the two hulls from the color image
            # Find bounding rects of original and shrunken hulls
            x, y, w, h = cv2.boundingRect(extraction_shape)
            x_eroded, y_eroded, w_eroded, h_eroded = cv2.boundingRect(cv2.findNonZero(eroded_hull_mask))

            # Adjust the coordinates for the cropped region if necessary
            x_crop = max(x, x_eroded)
            y_crop = max(y, y_eroded)
            w_crop = min(x + w, x_eroded + w_eroded) - x_crop
            h_crop = min(y + h, y_eroded + h_eroded) - y_crop

            # Crop the color image based on the bounding rectangle of the original hull
            cropped_image = color_image[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop]
            cropped_region_mask = region_mask[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop]
            cropped_image = cv2.bitwise_and(cropped_image, cropped_image, mask=cropped_region_mask)

            # Optional: Compute the bounding box for display or other purposes
            box = cv2.boxPoints(cv2.minAreaRect(hull))
            box = np.int0(box)
            box_detected = True
            
            #check midpoint
            height, width = color_image.shape[:2]
            if abs(y_center-(height/2)) > 30 or abs(x_center-(width/2)) > 70 :
                box_detected = False
                
    if box_detected == False:
        height, width = color_image.shape[:2]
        h = np.int(height/2)
        w = np.int(width/2)      
        cv2.circle(color_image, (w,h), 7, (0, 0, 200), 5)
        cv2.putText(color_image,f'No Bin Detected', (w-60,h-30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 200), 3)
        
        cv2.circle(masked_color_image, (w,h), 7, (0, 0, 200), 5)
        cv2.putText(masked_color_image,f'No Bin Detected', (w-60,h-30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 200), 3)
        
    if box_detected == True:
        height, width = color_image.shape[:2]
        h = np.int(height/2)
        w = np.int(width/2)      
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
cutting_depth = 0.7

try:
    while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
               
                continue
            
            filtered_depth_frame = temporal_filter.process(depth_frame) 
            filtered_depth_frame = spatial_filter.process(filtered_depth_frame)
            filtered_depth_frame = hole_filling_filter.process(filtered_depth_frame)
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(filtered_depth_frame.get_data())
            

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.25), cv2.COLORMAP_VIRIDIS)
           

            #masked_color_image, hull,box,box_detected = cut_region_v2(depth_image,color_image,min_depth = 0,max_depth = 0.8)
            masked_color_image,cropped_image, hull,box,box_detected = cut_region_between_hulls(depth_image,color_image,min_depth = 0,max_depth = cutting_depth, erosion_size_input= 12, cut_rect= True, improved_bounding_box= True)
            
            box_detected = False
            ####Add Hole Detection
            if box_detected == True:

                color_image = masked_color_image
        
        
            scale_factor = 0.4
            resized_masked_image = cv2.resize(masked_color_image,None,fx =  scale_factor,fy = scale_factor)
            resized_color_image = cv2.resize(color_image,None,fx =  scale_factor,fy = scale_factor)
            resized_depth_image = cv2.resize(depth_colormap,None,fx =  scale_factor,fy = scale_factor)

            # Create the top horizontal stack
            top_row = np.hstack((resized_masked_image, resized_color_image))

            
    
            if cv2.waitKey(1) & 0xFF == ord('q') :
                break
            
                # Display the images
            cv2.imshow('Color Images', top_row)
            #cv2.imshow('Color Image', resized_color_image)
            cv2.imshow('Depth Image', resized_depth_image)
            #cv2.imshow('masked Image',masked_color_image)

                
    
except RuntimeError as e:
    print(f"Capture stopped: {e}")

finally:
    # Stop streaming
    #pipeline.stop()
    
    print('Bincontroll stopped')

    cv2.destroyAllWindows()
    
        
        
