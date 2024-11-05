import numpy as np
import pyrealsense2 as rs
import cv2

from video_capture import capture_frames
from roi_functions import get_corner_points, get_shifted_points
import queue
import time
import threading
import math

import os
import re

# Define base directory
hull_dir = "dataset/hull"
factor_10_dir = "dataset/factor10"
factor_12_dir = "dataset/factor12"
factor_15_dir = "dataset/factor15"
hole_dir = "dataset/holes"
color_image_dir = "dataset/color_image"

image_class = 'positive'

# Create 'positive' and 'negative' directories if they don't exist
os.makedirs(os.path.join(hull_dir, "positive"), exist_ok=True)
os.makedirs(os.path.join(hull_dir, "negative"), exist_ok=True)
os.makedirs(os.path.join(factor_10_dir, "positive"), exist_ok=True)
os.makedirs(os.path.join(factor_10_dir, "negative"), exist_ok=True)
os.makedirs(os.path.join(factor_12_dir, "positive"), exist_ok=True)
os.makedirs(os.path.join(factor_12_dir, "negative"), exist_ok=True)
os.makedirs(os.path.join(factor_15_dir, "positive"), exist_ok=True)
os.makedirs(os.path.join(factor_15_dir, "negative"), exist_ok=True)
os.makedirs(os.path.join(hole_dir, "positive"), exist_ok=True)
os.makedirs(os.path.join(hole_dir, "negative"), exist_ok=True)
os.makedirs(os.path.join(color_image_dir, "positive"), exist_ok=True)
os.makedirs(os.path.join(color_image_dir, "negative"), exist_ok=True)

def get_highest_image_index(folder_path, class_name):
    """
    Scans the specified folder for existing images of a given class and
    returns the highest index found. Assumes filenames are in the format 'class_index.jpg'.
    
    Parameters:
    - folder_path: Path to the folder containing images.
    - class_name: The class ('positive' or 'negative') whose images we want to scan.

    Returns:
    - The next index to start saving images from.
    """
    highest_index = 0
    pattern = re.compile(rf"{class_name}_(\d+)\.jpg")  # Regex to extract index from filenames
    
    # List all files in the class folder
    class_folder = os.path.join(folder_path, class_name)
    if not os.path.exists(class_folder):
        return highest_index  # Folder doesn't exist, so we start from 0

    for filename in os.listdir(class_folder):
        match = pattern.match(filename)
        if match:
            index = int(match.group(1))
            highest_index = max(highest_index, index)
    
    return highest_index + 1  # Start with the next index


def save_image(image, label, image_name,save_dir):
    """
    Saves an image to the specified label directory ('positive' or 'negative').

    Parameters:
    - image: The image data (numpy array).
    - label: 'positive' or 'negative'.
    - image_name: Name for the image file (e.g., 'image1.jpg').
    """
    # Save to the appropriate directory
    save_path = os.path.join(save_dir, label, image_name)
    cv2.imwrite(save_path, image)
    print(f"Image saved at {save_path}")


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

def draw_rotated_rectangle(color_image,x1,x2,y1,y2,p_new_1):
                angle = -np.degrees(np.arctan2(y2-y1, x2-x1))
                
                x, y = p_new_1[0], p_new_1[1]

                # Define the width and height of the rectangle
                width = 80
                height = 45

                rotation_matrix = cv2.getRotationMatrix2D((x, y), angle, 1)
                
                corner_1  =[x+width, y-height]
                corner_2 = [x+width, y+height]
                corner_3 = [x-width, y+height]
                corner_4 = [x-width,y-height]
                
                rect_points = np.array([
                corner_1,
                corner_2,
                corner_3,
                corner_4
            ])

                rotated_rect_points = np.dot(rect_points, rotation_matrix[:, :2].T) + rotation_matrix[:, 2]
                
                rotated_rect_points = rotated_rect_points.astype(int)

                rotated_rect_points[:,0] =  rotated_rect_points[:,0]  #+ x
                rotated_rect_points[:,1] =  rotated_rect_points[:,1]  #+ y
                
                

          

                rect = np.array(rect_points,dtype = np.float32)
                width = np.linalg.norm(rect_points[0] - rect_points[1])
                height = np.linalg.norm(rect_points[0] - rect_points[3])

                dst_pts = np.array([
                    [0, 0],
                    [width, 0],
                    [width, height],
                    [0, height]
                ], dtype="float32")

                # Compute the perspective transform matrix
                M = cv2.getPerspectiveTransform(rect, dst_pts)

                # Perform the perspective transformation (i.e., crop the image)
                cropped_image = cv2.warpPerspective(color_image, M, (int(width), int(height)))
                
                ##Save IMAGE
    

                # Optional: Display the result to verify
                cv2.imshow("Cropped Rectangle", cropped_image)
                cv2.polylines(color_image, [rotated_rect_points], isClosed=True, color=(125, 125, 0), thickness=10)
                
                
                return cropped_image



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

def cut_region_between_hulls(depth_image, color_image, min_depth=0, max_depth=0.8, shrink_factor=0.12, cut_rect = False, improved_bounding_box = False):
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
                
               
                
                #get rect midpoint
                rect_center = rect[0]
                x_center = rect_center[0]
                y_center = rect_center[1]
                
     
          
            # Get the two points with the smallest y-coordinates
            
            #end experiment

           
            extraction_shape = np.array(extraction_shape)
            shape = extraction_shape
            #extraction_shape = enlarge_contour(extraction_shape, factor)

            # Step 1: Shrink the contour points
            #shrink_factor = 0.12# 80% shrink (inward move)
            shrunk_contour = shrink_contour_stable(shape, min_move_ratio=0.05, factor=shrink_factor)
   
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
                if abs(y_center-(height/2)) > 50 or abs(x_center-(width/2)) > 30 :
                    box_detected = False
        
    if box_detected == False:
        height, width = color_image.shape[:2]
        h = np.int0(height/2)
        w = np.int0(width/2)      
        cv2.circle(color_image, (w,h), 7, (0, 0, 200), 5)
        cv2.putText(color_image,f'No Bin Detected', (w-60,h-30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 200), 3)
        
        #cv2.circle(masked_color_image, (w,h), 7, (0, 0, 200), 5)
        #cv2.putText(masked_color_image,f'No Bin Detected', (w-60,h-30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 200), 3)
        
    if box_detected == True:
        height, width = color_image.shape[:2]
        h = np.int0(height/2)
        w = np.int0(width/2)      
        cv2.circle(color_image, (w,h), 7, (0, 200, 0), 5)
        cv2.putText(color_image,f'Bin Detected', (w-60,h-30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 0), 3)
        
        #cv2.circle(masked_color_image, (w,h), 7, (0, 200, 0), 5)
        #cv2.putText(masked_color_image,f'Bin Detected', (w-60,h-30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 0), 3)


     

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

i = get_highest_image_index(factor_12_dir, image_class)
h =get_highest_image_index(hole_dir, image_class)

box_detected_old = False
box_detected_counter = 0

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
            masked_color_image,cropped_image, hull,box,box_detected = cut_region_between_hulls(depth_image,color_image,min_depth = 0,max_depth = cutting_depth, shrink_factor=0.15, cut_rect= True, improved_bounding_box= False)
            print('box detected counter ', box_detected_counter)
            if box_detected == True:
                 box_detected_counter +=1
            if box_detected == False:
                 box_detected_counter =0
            if box_detected == True and  box_detected_counter > 30:
                image_name = f'{image_class}_{i}.jpg'
                save_image(color_image,image_class, image_name, color_image_dir)
                
                box_detected_counter =0
                masked_color_image,cropped_image, hull,box,box_detected = cut_region_between_hulls(depth_image,color_image,min_depth = 0,max_depth = cutting_depth, shrink_factor=0.15, cut_rect= False, improved_bounding_box= False)
                
                save_image(masked_color_image,image_class, image_name, hull_dir)
                masked_color_image,cropped_image, hull,box,box_detected = cut_region_between_hulls(depth_image,color_image,min_depth = 0,max_depth = cutting_depth, shrink_factor=0.1, cut_rect= True, improved_bounding_box= False)
                save_image(masked_color_image,image_class, image_name, factor_10_dir)
                masked_color_image,cropped_image, hull,box,box_detected = cut_region_between_hulls(depth_image,color_image,min_depth = 0,max_depth = cutting_depth, shrink_factor=0.12, cut_rect= True, improved_bounding_box= False)
                save_image(masked_color_image,image_class, image_name, factor_12_dir)
                masked_color_image,cropped_image, hull,box,box_detected = cut_region_between_hulls(depth_image,color_image,min_depth = 0,max_depth = cutting_depth, shrink_factor=0.15, cut_rect= True, improved_bounding_box= False)
                save_image(masked_color_image,image_class, image_name, factor_15_dir)
                i+=1
                save_holes = True
                if save_holes == True:
                        corner_1, corner_2, corner_3, corner_4 = get_corner_points(color_image, box, hull)
                    
                        edge_scale_factor = 0.2
        
                        p_new_1, p_new_2, p_new_3, p_new_4, d1,d2, x1,y1,x2,y2,x3,y3,x4,y4 = get_shifted_points(edge_scale_factor,corner_1,corner_2,corner_3, corner_4)

                        ########## draw rectangle
                     
                        hole_1 = draw_rotated_rectangle(color_image,x1,x2,y1,y2,p_new_1)
                        save_image(hole_1,image_class, f'{image_class}_{h}.jpg', hole_dir)
                        h+=1
                        hole_2 = draw_rotated_rectangle(color_image,x1,x2,y1,y2,p_new_2)
                        save_image(hole_2,image_class, f'{image_class}_{h}.jpg', hole_dir)
                        h+=1
                        hole_3 = draw_rotated_rectangle(color_image,x3,x4,y3,y4,p_new_3)
                        save_image(hole_3,image_class, f'{image_class}_{h}.jpg', hole_dir)
                        h+=1
                        hole_4 = draw_rotated_rectangle(color_image,x3,x4,y3,y4,p_new_4)
                        save_image(hole_4,image_class, f'{image_class}_{h}.jpg', hole_dir)
                        h+=1

            box_detected_old = box_detected
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
    
            if cv2.waitKey(1) & 0xFF == ord('q') :
                break

            

            
            
  

                
    
except RuntimeError as e:
    print(f"Capture stopped: {e}")

finally:
    # Stop streaming
    #pipeline.stop()
    
    print('Bincontroll stopped')

    cv2.destroyAllWindows()
    
        
        
