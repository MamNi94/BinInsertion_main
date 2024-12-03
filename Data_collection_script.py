import numpy as np
import pyrealsense2 as rs
import cv2
import random
import matplotlib.pyplot as plt

from video_capture import capture_frames
from roi_functions import get_corner_points, get_shifted_points
import queue
import time
import threading
import math

import os
import re

# Define base directory

dataset = 'dataset/data_0312_params_1015_102_088_101/'
wall_dir= f"{dataset}walls_only"
hole_dir = f"{dataset}holes"
color_image_dir = f"{dataset}color_image"
wall_dir_2 = f"{dataset}walls_thin"
wall_dir_3 = f"{dataset}walls_normal"



image_class_bin = 'positive'
image_class_hole = 'positive'

save_holes = False

# Create 'positive' and 'negative' directories if they don't exist

os.makedirs(os.path.join(wall_dir, "positive"), exist_ok=True)
os.makedirs(os.path.join(wall_dir, "negative"), exist_ok=True)


os.makedirs(os.path.join(hole_dir, "positive"), exist_ok=True)
os.makedirs(os.path.join(hole_dir, "negative"), exist_ok=True)
os.makedirs(os.path.join(color_image_dir, "positive"), exist_ok=True)
os.makedirs(os.path.join(color_image_dir, "negative"), exist_ok=True)
os.makedirs(os.path.join(wall_dir_2, "positive"), exist_ok=True)
os.makedirs(os.path.join(wall_dir_2, "negative"), exist_ok=True)
os.makedirs(os.path.join(wall_dir_3, "positive"), exist_ok=True)
os.makedirs(os.path.join(wall_dir_3, "negative"), exist_ok=True)

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


def divide_into_quadrants(corners,color_image):
    # Calculate the bounding box center
     # Get the dimensions of the color image
    img_height, img_width, _ = color_image.shape
    
    # Calculate the center of the color image
    center_x = img_width / 2
    center_y = img_height / 2

    # Classify points into quadrants
    quadrants = {"top_left": [], "top_right": [], "bottom_left": [], "bottom_right": []}
    for x, y in corners:
        if x < center_x and y > center_y:
            quadrants["top_left"].append((x, y))
        elif x >= center_x and y > center_y:
            quadrants["top_right"].append((x, y))
        elif x < center_x and y <= center_y:
            quadrants["bottom_left"].append((x, y))
        elif x >= center_x and y <= center_y:
            quadrants["bottom_right"].append((x, y))
    return quadrants

def find_lines(corners,color_image):
    quadrants = divide_into_quadrants(corners,color_image)

    # Find two points with min_x in left two quadrants
    left_points = quadrants["top_left"] + quadrants["bottom_left"]
    min_x_points = sorted(left_points, key=lambda p: p[0])[:2]

    # Find two points with max_x in right two quadrants
    right_points = quadrants["top_right"] + quadrants["bottom_right"]
    max_x_points = sorted(right_points, key=lambda p: p[0], reverse=True)[:2]

    # Find two points with min_y in upper two quadrants
    upper_points = quadrants["top_left"] + quadrants["top_right"]
    min_y_points = sorted(upper_points, key=lambda p: p[1],reverse= True)[:2]

    # Find two points with max_y in lower two quadrants
    lower_points = quadrants["bottom_left"] + quadrants["bottom_right"]
    max_y_points = sorted(lower_points, key=lambda p: p[1], reverse=False)[:2]

    # Create lines
    lines = {
        "min_x_line": (min_x_points[0], min_x_points[1]),
        "max_x_line": (max_x_points[0], max_x_points[1]),
        "min_y_line": (min_y_points[0], min_y_points[1]),
        "max_y_line": (max_y_points[0], max_y_points[1]),
    }
    return lines


def compute_line_equation(point1, point2):
    """Get line equation coefficients A, B, C for line through point1 and point2."""
    x1, y1 = point1
    x2, y2 = point2
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return A, B, C

def find_intersection(line1, line2):
    """Find the intersection point of two lines, if it exists."""
    A1, B1, C1 = line1
    A2, B2, C2 = line2

    determinant = A1 * B2 - A2 * B1
    if determinant == 0:
        return None  # Parallel lines, no intersection

    x = (B1 * C2 - B2 * C1) / determinant
    y = (A2 * C1 - A1 * C2) / determinant
    return (x, y)

def compute_intersections(lines, color_image):
    """Compute intersection points of all lines."""
    line_equations = [compute_line_equation(*line) for line in lines.values()]
    intersection_points = []
     # Get the dimensions of the color image
    img_height, img_width, _ = color_image.shape
    
    # Calculate the image boundaries
    min_x, min_y = 0, 0
    max_x, max_y = img_width, img_height


    # Find intersections of all line pairs
    for i in range(len(line_equations)):
        for j in range(i + 1, len(line_equations)):
            intersection = find_intersection(line_equations[i], line_equations[j])
            if intersection:
                #intersection_points.append(intersection)
                #intersection_points.append([intersection[0], intersection[1]])
                x, y = intersection[0], intersection[1]

                # Check if the intersection is within the image boundaries
                if min_x <= x <= max_x and min_y <= y <= max_y:
                    intersection_points.append([int(x), int(y)])
      

    return intersection_points


def cross(o, a, b):
    """Cross product of vectors OA and OB. A positive cross product indicates a counter-clockwise turn, 0 indicates a collinear point, and negative indicates a clockwise turn."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def order_points(points):
    """Order points in counter-clockwise direction to form the convex hull."""
    # Sort the points based on x, then y
    points = sorted(points, key=lambda p: (p[0], p[1]))

    # Build the lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build the upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Remove the last point of each half because it's repeated at the beginning of the other half
    return lower[:-1] + upper[:-1]


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

i = get_highest_image_index(wall_dir, image_class_bin)
h =get_highest_image_index(hole_dir, image_class_hole)

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
            K =21
            kernel = np.ones((K, K), np.uint8)  # You can adjust the kernel size

            depth_image = cv2.morphologyEx(depth_image, cv2.MORPH_CLOSE, kernel)
            
           
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.25), cv2.COLORMAP_VIRIDIS)

            cut_region_final = None

            scale_factor = 0.5
            resized_depth_image = cv2.resize(depth_colormap,None,fx =  scale_factor,fy = scale_factor)
            cv2.imshow('Depth Image_', resized_depth_image)
            #masked_color_image, hull,box,box_detected = cut_region_v2(depth_image,color_image,min_depth = 0,max_depth = 0.8)
            hull,box,box_detected = detect_box(depth_image,color_image,min_depth = 0,max_depth = cutting_depth)
            print('box detected counter ', box_detected_counter)
            if box_detected == True:
                 box_detected_counter +=1
            if box_detected == False:
                 box_detected_counter =0
            if box_detected == True and  box_detected_counter > 2:
                box_detected_counter = 0
                image_name = f'{image_class_bin}_{i}.jpg'
               
                

                corners = get_corner_points(color_image, box, hull)
                
                #get corner points
                #lines = find_lines(corners,color_image)

                # Usage example

                #intersection_points = compute_intersections(lines,color_image)
                #end get corner points

                # Display the modified image
                #cv2.imshow("Lines Overlay", color_image)
                save_image(color_image,image_class_bin, image_name, color_image_dir)
                    

                ###cut bin
                hull = cv2.convexHull(np.array(corners))

                # Find the centroid of the hull
                centroid = np.mean(hull[:, 0, :], axis=0)

                def get_region(scale_factor_outer = 1.015, adjustent_factor_outer = 1.015, scale_factor_inner = 0.86, adjustent_factor_inner = 0.86):

            
                    # Expand the hull outward
                    expanded_hull = scale_hull(hull, scale_factor_outer, adjustent_factor_outer)
                
                    shrunk_hull = scale_hull(hull, scale_factor_inner,adjustent_factor_inner)


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
                    return cut_region_final
                

                scale_factor_out = random.uniform(1.025,1.010)
                scale_factor_inner = random.uniform(0.88,0.8)
                scale_factor_out_djustment = random.uniform(1.02,1.01)
                scale_factor_inner_adjustment = random.uniform(1.02,1.01)

                #cut_region_final_1 = get_region(scale_factor_outer = 1.04, adjustent_factor_outer = 1.02, scale_factor_inner = 0.82, adjustent_factor_inner = 1.01)
                cut_region_final_3 = get_region(scale_factor_outer = 1.015, adjustent_factor_outer = 1.02, scale_factor_inner = 0.86, adjustent_factor_inner = 1.01)
                #cut_region_final_3 = get_region(scale_factor_outer = scale_factor_out, adjustent_factor_outer = scale_factor_out_djustment, scale_factor_inner = scale_factor_inner, adjustent_factor_inner =scale_factor_inner_adjustment)
                cut_region_final_1 = get_region(scale_factor_outer = 0.95, adjustent_factor_outer = 1.02, scale_factor_inner = 0.86, adjustent_factor_inner = 1.01)
                cut_region_final_2 = get_region(scale_factor_outer = 1.01, adjustent_factor_outer = 1.02, scale_factor_inner = 0.88, adjustent_factor_inner = 1.01)

                edge_scale_factor = 0.19

                p_new_1, p_new_2, p_new_3, p_new_4, d1,d2, x1,y1,x2,y2,x3,y3,x4,y4 = get_shifted_points(edge_scale_factor,corners)
                
               
                if save_holes == True:
                        

                        ########## draw rectangle
                     
                        hole_1 = draw_rotated_rectangle(color_image,x1,x2,y1,y2,p_new_1)
                        save_image(hole_1,image_class_hole, f'{image_class_hole}_{h}.jpg', hole_dir)
                        h+=1
                        hole_2 = draw_rotated_rectangle(color_image,x1,x2,y1,y2,p_new_2)
                        save_image(hole_2,image_class_hole, f'{image_class_hole}_{h}.jpg', hole_dir)
                        h+=1
                        hole_3 = draw_rotated_rectangle(color_image,x3,x4,y3,y4,p_new_3)
                        save_image(hole_3,image_class_hole, f'{image_class_hole}_{h}.jpg', hole_dir)
                        h+=1
                        hole_4 = draw_rotated_rectangle(color_image,x3,x4,y3,y4,p_new_4)
                        save_image(hole_4,image_class_hole, f'{image_class_hole}_{h}.jpg', hole_dir)
                        h+=1
                
                save_walls = True
                if save_walls == True:
                    scale_factor = 0.4
                    cut_region_final_small = cv2.resize(cut_region_final_2,None, fx = scale_factor, fy= scale_factor)
                    cv2.imshow('cut out',cut_region_final_small)     
                    save_image(cut_region_final_1,image_class_bin, image_name, wall_dir)
                    save_image(cut_region_final_2,image_class_bin, image_name, wall_dir_2)
                    save_image(cut_region_final_3,image_class_bin, image_name, wall_dir_3)
                    i = i+1
            box_detected_old = box_detected
            scale_factor = 0.4
            resized_color_image = cv2.resize(color_image,None,fx =  scale_factor,fy = scale_factor)
            
            #print(masked_color_image)
            if cut_region_final is not None and   isinstance(cut_region_final, np.ndarray):
               
                resized_masked_image = cv2.resize(cut_region_final,None,fx =  scale_factor,fy = scale_factor)
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
    
        
        
