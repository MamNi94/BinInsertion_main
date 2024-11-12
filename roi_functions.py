import numpy as np
import cv2
import math



def draw_rectangle(image, barcode):
    x, y, w, h = barcode.rect
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    barcode_data = barcode.data.decode('utf-8')
    print(barcode_data)
    cv2.putText(image, barcode_data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def angle_between(p1, p2, p3):
    """Calculate the angle between the vectors (p1-p2) and (p3-p2) in degrees."""
    v1 = p1 - p2
    v2 = p3 - p2
    angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0])
    return np.abs(np.degrees(angle)) % 180

def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def move_point_on_line(p1, p2, point, distance):
    """
    Move a point along the line defined by p1 and p2 by a certain distance.
    
    Parameters:
    p1 (tuple): First point defining the line (x1, y1).
    p2 (tuple): Second point defining the line (x2, y2).
    point (tuple): The point to move (x, y).
    distance (float): The distance to move along the line.
    
    Returns:
    tuple: New coordinates of the point.
    """
    
    # Convert points to numpy arrays
    p1 = np.array(p1)
    p2 = np.array(p2)
    point = np.array(point)
    
    # Calculate direction vector of the line
    direction = p2 - p1
    
    # Normalize the direction vector
    norm = np.linalg.norm(direction)
    direction_normalized = direction / norm
    
    # Calculate the new point
    new_point = point + direction_normalized * distance
   
    new_x = int(new_point[0])
    new_y = int(new_point[1])
    
    return tuple([new_x,new_y])


def cut_region_at_distance(depth_image,color_image,min_depth = 0,max_depth = 0.8, region = False):
                if region == False:
                    mask = np.logical_and(depth_image > min_depth * 1000, depth_image < max_depth * 1000)

                    # Convert mask to uint8
                    mask = mask.astype(np.uint8) * 255
                    masked_color_image = cv2.bitwise_and(color_image, color_image, mask=mask)
                    return masked_color_image

                if region == True:
                     
                    min_depth_mm = min_depth * 1000
                    max_depth_mm = max_depth * 1000

                    # Create a mask for the specified depth range
                    mask = np.logical_and(depth_image > min_depth_mm, depth_image < max_depth_mm)

                    # Convert mask to uint8
                    mask = mask.astype(np.uint8) * 255

                    # Find contours in the mask
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # Create an empty mask to draw the contour region
                    contour_mask = np.zeros_like(mask)
                      # Check if any contours were found
                    if contours:
                        all_points = np.concatenate(contours)
                        hull = cv2.convexHull(all_points)
                        # Draw the filled contour on the contour_mask
                        cv2.drawContours(contour_mask, [hull], -1, (255), thickness=cv2.FILLED)

                    masked_color_image = cv2.bitwise_and(color_image, color_image, mask=contour_mask)

                    return masked_color_image
                
                
def detect_roi(binary_thresh,color_image, min_angle = 70, max_angle = 110, detect_center = True):
        box_detected = False
        
        center = [0,0]
        bbox = [[[0,0],[0,0],[0,0],[0,0]]]
      
        # Find contours
        contours, _ = cv2.findContours(binary_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        # Iterate through contours and find rectangles
        rectangles = []
       
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            bbox = approx
            # Check if contour has 4 vertices (is rectangle)
            if len(approx) == 4:
        
                angles = []
                for i in range(4):
                    p1 = approx[i][0]
                    p2 = approx[(i + 1) % 4][0]
                    p3 = approx[(i + 2) % 4][0]
                    angle = angle_between(p1, p2, p3)
                    angles.append(angle)
                

                if all(min_angle < angle < max_angle for angle in angles):
                    rectangles.append(approx)
                    
                    if detect_center == True:
                       
                        # Calculate the center of the rectangle
                        M = cv2.moments(approx)
                        if M["m00"] != 0:
                            center_x = int(M["m10"] / M["m00"])
                            center_y = int(M["m01"] / M["m00"])
                        else:
                            center_x, center_y = 0, 0

                        height, width = color_image.shape[:2]
                        d = np.sqrt((center_x-(width/2))**2 + (center_y-(height/2))**2)

                    
                        # Draw the center
                        if d <=100:
                            cv2.circle(color_image, (center_x, center_y), 7, (0, 0, 255), 5)
                            cv2.putText(color_image,'Bin Detected', (center_x-20, center_y-50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 5 )
                            #cv2.drawContours(color_image, [approx], -1, (0, 255, 0),10)
                            box_detected = True
                            center = [center_x,center_y]
                            bbox = approx
                            return  box_detected, center, color_image, bbox
                    
                    if detect_center == False:
                        
                        cv2.drawContours(color_image, [approx], -1, (0, 255, 255),10)
                        box_detected = True
                    
        return  box_detected, center, color_image, bbox
    
    
    
def cut_region_v2(depth_image,color_image,min_depth = 0,max_depth = 0.8):

        masked_color_image, hull,box = 0,0,0
        box_detected = False
        min_depth_mm = min_depth * 1000
        max_depth_mm = max_depth * 1000

        # Create a mask for the specified depth range
        mask = np.logical_and(depth_image > min_depth_mm, depth_image < max_depth_mm)

        # Convert mask to uint8
        mask = mask.astype(np.uint8) * 255

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Create an empty mask to draw the contour region
        contour_mask = np.zeros_like(mask)
            # Check if any contours were found
        if contours:
            all_points = np.concatenate(contours)
            hull = cv2.convexHull(all_points)
            # Draw the filled contour on the contour_mask
            cv2.drawContours(contour_mask, [hull], -1, (255), thickness=cv2.FILLED)

            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            #cv2.drawContours(color_image, [box], 0, (125, 125, 255), 5)
            #cv2.drawContours(color_image, [hull], 0, (0, 255, 0), 5)
            box_detected = True
            
        masked_color_image = cv2.bitwise_and(color_image, color_image, mask=contour_mask)
                    
        return masked_color_image, hull,box, box_detected

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

def improve_bounding_box(color_image,box):
    for i in range(4):
        #p1 = box[i - 1]  # Previous point (wraps around using i-1)
        p2 = box[i]      # Current point
       # p3 = box[(i + 1) % 4]  # Next point (wraps around using i+1)
        
        #angle = np.round(calculate_angle(p1, p2, p3))
        cv2.putText(color_image,f'corner: {i}', p2, cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 200), 3)
                

    max_1, max_2 = get_max_points(box)
    print(max_1, max_2)
    
    bin_side_length = np.round(distance(max_1,max_2))
        
    cv2.putText(color_image,f'p1', (max_1[0], max_1[1]+30) , cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 0), 3)
    cv2.putText(color_image,f'p2', (max_2[0],max_2[1] + 30) , cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 0), 3)
    cv2.putText(color_image,f'right wall length: {bin_side_length}', (np.int0(max_2[0])-100,np.int0((max_2[1]+max_1[1])/2) ) , cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 0), 3)

    if max_1[1] > max_2[1]:
        direction = np.array([-(max_1[1]-max_2[1])/bin_side_length,(max_1[0] - max_2[0])/bin_side_length])
    else:
        direction = np.array([(max_1[1]-max_2[1])/bin_side_length,-(max_1[0] - max_2[0])/bin_side_length])
    calculated_point_1 = direction * 1.45 * bin_side_length  + max_1
    calculated_point_2 = direction * 1.45 * bin_side_length  + max_2
    
    cv2.circle(color_image, (np.int0(calculated_point_1[0]),np.int0(calculated_point_1[1])), 3,(255,255,0),3)
    cv2.circle(color_image, (np.int0(calculated_point_2[0]),np.int0(calculated_point_2[1])), 3,(255,255,0),3)
    
    print(calculated_point_1, calculated_point_2, max_1,max_2)
    all_points = np.array([np.int0(calculated_point_1), np.int0(calculated_point_2), max_1,max_2])
    rect = cv2.minAreaRect(all_points)
    box = cv2.boxPoints(rect)  # Get the four vertices of the rectangle
    box = np.int0(box) 
  
    return box

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


def cut_region_between_hulls_v2(depth_image, color_image, min_depth=0, max_depth=0.8,  cut_rect = False, improved_bounding_box = False):
    masked_color_image, cropped_image, hull, box = 0, 0, 0, 0
    box_detected = False
    min_depth_mm = min_depth * 1000
    max_depth_mm = max_depth * 1000

    # Step 1: Create a binary mask based on the depth range
    mask = np.logical_and(depth_image > min_depth_mm, depth_image < max_depth_mm)

    # Step 2: Convert the mask to uint8 for further processing
    mask = mask.astype(np.uint8) * 255

    if mask.dtype != np.uint8:
     mask = cv2.convertScaleAbs(mask, alpha=(255.0 / mask.max()))

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
                
                ####Improved Bounding Box
                if improved_bounding_box ==True:
                    for i in range(4):
                        p = box[i]      # Current point
                    
                        #cv2.putText(color_image,f'corner: {i}', p, cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 200), 3)                

                    max_1, max_2 = get_max_points(box)
                 
                
                    bin_side_length = np.round(distance(max_1,max_2))
                        
                  
                    if max_1[1] > max_2[1]:
                        direction = np.array([-(max_1[1]-max_2[1])/bin_side_length,(max_1[0] - max_2[0])/bin_side_length])
                    else:
                        direction = np.array([(max_1[1]-max_2[1])/bin_side_length,-(max_1[0] - max_2[0])/bin_side_length])
               
                    bin_factor = 1.45 #1.45
                    #bin_factor = 1/1.45
                    calculated_point_1 = (direction * bin_factor * bin_side_length)  + max_1
                    calculated_point_2 = (direction * bin_factor * bin_side_length)  + max_2
      
                    
                    print(calculated_point_1, calculated_point_2, max_1,max_2)
                    all_points = np.array([np.int0(calculated_point_1), np.int0(calculated_point_2), max_2,max_1])
                    box = cv2.convexHull(all_points)
                 
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
            print('hi')
            print('region mask ', region_mask)
            if region_mask.any():
                masked_color_image = cv2.bitwise_and(color_image, color_image, mask=region_mask)
                print('masked',masked_color_image)
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
    
    
def cut_region_between_hulls(depth_image, color_image, min_depth=0, max_depth=0.8,  cut_rect = False, improved_bbox = False):
    masked_color_image, cropped_image, hull, box = 0, 0, 0, 0
    box_detected = False
    min_depth_mm = min_depth * 1000
    max_depth_mm = max_depth * 1000

    # Step 1: Create a binary mask based on the depth range
    mask = np.logical_and(depth_image > min_depth_mm, depth_image < max_depth_mm)

    # Step 2: Convert the mask to uint8 for further processing
    mask = mask.astype(np.uint8) * 255

    # Step 3: Optional - Clean the mask using morphological operations
    #kernel = np.ones((5, 5), np.uint8)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close small holes
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove small noise

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

            extraction_shape = hull
            
            if cut_rect == True:
                rect = cv2.minAreaRect(all_points)
                box = cv2.boxPoints(rect)  # Get the four vertices of the rectangle
                box = np.int0(box) 
                extraction_shape = box
                print(extraction_shape)
                if improved_bbox ==True:
                     extraction_shape = improve_bounding_box(color_image, box)
                
                #get rect midpoint
                rect_center = rect[0]
                x_center = rect_center[0]
                y_center = rect_center[1]
                print('center', x_center,y_center)
                    
            '''
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
            '''
            factor = 0.85  # 80% shrink (inward move)
            extraction_shape = np.array(extraction_shape)

            # Step 1: Shrink the contour points
            shrunk_contour = shrink_contour(extraction_shape, factor)

            # Step 2: Create a mask and draw the shrunk contour
            hull_mask = np.zeros_like(mask)
            cv2.drawContours(hull_mask, [extraction_shape], -1, 255, thickness=cv2.FILLED)
            shrunk_mask = np.zeros_like(mask)  # Empty mask for the shrunk contour
            cv2.drawContours(shrunk_mask, [shrunk_contour], -1, 255, thickness=cv2.FILLED)  # Shrunk filled shape
        
            
            # Step 9: Compute the mask for the region between the original and shrunken hulls
            region_mask = cv2.bitwise_and(hull_mask, cv2.bitwise_not(shrunk_mask))

            # Step 10: Apply the region mask to the color image
            masked_color_image = cv2.bitwise_and(color_image, color_image, mask=region_mask)
            #cv2.imshow('masked_color_image', masked_color_image)

            box_detected = True
            
            #check midpoint
            height, width = color_image.shape[:2]
            if abs(y_center-(height/2)) > 30 or abs(x_center-(width/2)) > 70 :
                box_detected = False
                
    if box_detected == False:
        height, width = color_image.shape[:2]
        h = np.int0(height/2)
        w = np.int0(width/2)      
        cv2.circle(color_image, (w,h), 7, (0, 0, 200), 5)
        cv2.putText(color_image,f'No Bin Detected', (w-60,h-30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 200), 3)
        
    if box_detected == True:
        height, width = color_image.shape[:2]
        h = np.int0(height/2)
        w = np.int0(width/2)      
        cv2.circle(color_image, (w,h), 7, (0, 200, 0), 5)
        cv2.putText(color_image,f'Bin Detected', (w-60,h-30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 0), 3)
            

    return masked_color_image, cropped_image, hull, box, box_detected

def get_shifted_points(edge_scale_factor,corner_1,corner_2,corner_3, corner_4, offset = 20):
        side_lengths = [
                        (distance(corner_1, corner_2), (corner_1, corner_2)),
                        (distance(corner_2, corner_3), (corner_2, corner_3)),
                        (distance(corner_3, corner_4), (corner_3, corner_4)),
                        (distance(corner_4, corner_1), (corner_4, corner_1)),
                    ]

        #Box Stuff
        # Calculate the lengths of the sides

        # Sort sides by length in descending order and get the two longest sides
        longest_sides = sorted(side_lengths, key=lambda x: x[0], reverse=True)[:2]

        # Extract the lengths and coordinates of the two longest sides

        longest_sides_coords = [side[1] for side in longest_sides]
        
        x1 = longest_sides_coords[0][0][0]
        y1 = longest_sides_coords[0][0][1]
        
        x2 = longest_sides_coords[0][1][0]
        y2 = longest_sides_coords[0][1][1]
        
        x3 = longest_sides_coords[1][0][0]
        y3 = longest_sides_coords[1][0][1]
        
        x4 = longest_sides_coords[1][1][0]
        y4 = longest_sides_coords[1][1][1]


        y_values = [y1, y2, y3, y4]

        # Calculate the average of the y-values
        average_y = sum(y_values) / len(y_values)

        # Apply adjustments based on whether the value is above or below the average
        adjusted_y_values = []

        for y in y_values:
            if y < average_y:
                adjusted_y = y + offset  # Subtract the offset if the value is below the average
            else:
                adjusted_y = y -offset  # Add the offset if the value is above the average
            adjusted_y_values.append(adjusted_y)


        y1,y2,y3,y4 = adjusted_y_values[0], adjusted_y_values[1], adjusted_y_values[2],adjusted_y_values[3]
        
        p1, p2  = [x1,y1], [x2,y2]
        p3,p4 = [x3,y3], [x4,y4]
        

        d1 = calculate_distance(p1,p2)
        d2 = calculate_distance(p3,p4)
        
        
        edge_scale_factor = 0.20

            
        p_new_1 = move_point_on_line(p1, p2, p1, d1 * edge_scale_factor)
        p_new_2 = move_point_on_line(p1, p2, p2, -d1* edge_scale_factor)
        p_new_3 = move_point_on_line(p3,p4, p3, d2*edge_scale_factor)
        p_new_4 = move_point_on_line(p3,p4,p4,d2*-edge_scale_factor )
        
        return p_new_1, p_new_2, p_new_3, p_new_4, d1,d2, x1,y1,x2,y2,x3,y3,x4,y4
    
    
def calculate_distance(p1, p2):
    """ Calculate the Euclidean distance between two points. """
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_distance_scaled(p1, p2, x_scale,y_scale):
    """ Calculate the Euclidean distance between two points. """
    return np.sqrt(((p1[0]-p2[0])**2)*x_scale + ((p1[1]-p2[1])**2) * y_scale)
    
    
def get_corner_points(color_image,box,hull):
                x_corner_1 = 0
                y_corner_1 = 0
                reference_distance_corener_1 = 5000
                
                x_corner_2 = 0
                y_corner_2 = 0
                reference_distance_corener_2 = 5000
                
                x_corner_3 = 0
                y_corner_3 = 0
                reference_distance_corener_3 = 5000
                
                x_corner_4 = 0
                y_corner_4 = 0
                reference_distance_corener_4 = 5000
                
                for j in range(len(hull)):
                    p = tuple(hull[j][0])
                    x = p[0]
                    y = p[1]
                    

                    distance_corner_1 = calculate_distance_scaled(box[0], p, x_scale = 1, y_scale = 5)
                    distance_corner_2 = calculate_distance_scaled(box[1], p,x_scale = 1, y_scale = 5)
                    distance_corner_3 = calculate_distance_scaled(box[2], p,x_scale = 1, y_scale = 5)
                    distance_corner_4 = calculate_distance_scaled(box[3], p,x_scale = 1, y_scale = 5)
                    
                    if distance_corner_1 < reference_distance_corener_1:
                        x_corner_1 = x
                        y_corner_1 = y
                        reference_distance_corener_1 = distance_corner_1
                        
                    if distance_corner_2 < reference_distance_corener_2:
                        x_corner_2 = x
                        y_corner_2 = y
                        reference_distance_corener_2 = distance_corner_2
                    
                    if distance_corner_3 < reference_distance_corener_3:
                        x_corner_3 = x
                        y_corner_3 = y
                        reference_distance_corener_3 = distance_corner_3
                        
                      
                    if distance_corner_4 < reference_distance_corener_4:
                        x_corner_4 = x
                        y_corner_4 = y
                        reference_distance_corener_4 = distance_corner_4


                #cv2.circle(color_image, box[1], 20, (255, 0, 255), -1) 
                cv2.circle(color_image, [x_corner_1,y_corner_1], 20, (125, 0, 125), -1) 
                cv2.circle(color_image, [x_corner_2,y_corner_2], 20, (125, 0, 125), -1) 
                cv2.circle(color_image, [x_corner_3,y_corner_3], 20, (125, 0, 125), -1) 
                cv2.circle(color_image, [x_corner_4,y_corner_4], 20, (125, 0, 125), -1) 
                
                
        
                
                corner_1 = [x_corner_1, y_corner_1]
                corner_2 = [x_corner_2, y_corner_2]
                corner_3 = [x_corner_3, y_corner_3]
                corner_4 = [x_corner_4, y_corner_4]
                
                return corner_1, corner_2, corner_3, corner_4
            


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
                
                cv2.polylines(color_image, [rotated_rect_points], isClosed=True, color=(125, 125, 0), thickness=10)
                
                return rect_points