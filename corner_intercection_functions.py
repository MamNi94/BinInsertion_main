#corner intercection functions
import numpy as np
import cv2
def get_lines_from_corners(corners, centroid):
                        # Convert corners to a NumPy array if it's a list
                        corners = np.array(corners)

                        # If corners have an extra dimension (like (N, 1, 2)), flatten it to (N, 2)
                        if corners.ndim == 3:
                            corners = corners[:, 0, :]

                        # Now corners is a (N, 2) array
                        # Divide the points into quadrants
                        top_left = corners[(corners[:, 0] < centroid[0]) & (corners[:, 1] > centroid[1])]
                        top_right = corners[(corners[:, 0] > centroid[0]) & (corners[:, 1] > centroid[1])]
                        bottom_left = corners[(corners[:, 0] < centroid[0]) & (corners[:, 1] < centroid[1])]
                        bottom_right = corners[(corners[:, 0] > centroid[0]) & (corners[:, 1] < centroid[1])]

                        # Line 1: Smallest y value in bottom-left and bottom-right quadrants
                        bottom_left_min_y = bottom_left[np.argmin(bottom_left[:, 1])] if len(bottom_left) > 0 else None
                        bottom_right_min_y = bottom_right[np.argmin(bottom_right[:, 1])] if len(bottom_right) > 0 else None

                        # Line 2: Largest y value in top-left and top-right quadrants
                        top_left_max_y = top_left[np.argmax(top_left[:, 1])] if len(top_left) > 0 else None
                        top_right_max_y = top_right[np.argmax(top_right[:, 1])] if len(top_right) > 0 else None

                          # Line 3: Smallest x value in bottom-left and top-left quadrants
                        bottom_left_min_x = bottom_left[np.argmin(bottom_left[:, 0])] if len(bottom_left) > 0 else None
                        top_left_min_x = top_left[np.argmin(top_left[:, 0])] if len(top_left) > 0 else None

                        # Line 4: Largest x value in bottom-right and top-right quadrants
                        bottom_right_max_x = bottom_right[np.argmax(bottom_right[:, 0])] if len(bottom_right) > 0 else None
                        top_right_max_x = top_right[np.argmax(top_right[:, 0])] if len(top_right) > 0 else None

                        lines = []

                        if bottom_left_min_y is not None and bottom_right_min_y is not None:
                            # Add line 1: from bottom-left to bottom-right
                            lines.append((tuple(bottom_left_min_y), tuple(bottom_right_min_y)))

                        if top_left_max_y is not None and top_right_max_y is not None:
                            # Add line 2: from top-left to top-right
                            lines.append((tuple(top_left_max_y), tuple(top_right_max_y)))

                         # Line 3: from bottom-left to top-left (smallest x in left quadrants)
                        if bottom_left_min_x is not None and top_left_min_x is not None:
                            lines.append((tuple(bottom_left_min_x), tuple(top_left_min_x)))

                        # Line 4: from bottom-right to top-right (largest x in right quadrants)
                        if bottom_right_max_x is not None and top_right_max_x is not None:
                            lines.append((tuple(bottom_right_max_x), tuple(top_right_max_x)))


                        return lines


def extend_line(p1, p2, img_width, img_height):
                        """
                        Extend the line defined by points p1 and p2 to the image boundaries.
                        Returns the extended line as two points.
                        """
                        x1, y1 = p1
                        x2, y2 = p2
                        
                        # Parametric equations to extend the line
                        # We are extending in both directions, so t ranges from -inf to +inf
                        # Find the t-values where the line intersects the left, right, top, and bottom boundaries
                        
                        # Left boundary (x = 0)
                        if x2 != x1:  # Avoid division by zero for vertical lines
                            t_left = -x1 / (x2 - x1)
                            left_point = parametric_line(x1, y1, x2, y2, t_left)
                        else:
                            left_point = (0, y1)  # For vertical lines, use the x = 0 line directly
                        
                        # Right boundary (x = img_width)
                        if x2 != x1:  # Avoid division by zero for vertical lines
                            t_right = (img_width - x1) / (x2 - x1)
                            right_point = parametric_line(x1, y1, x2, y2, t_right)
                        else:
                            right_point = (img_width, y1)  # For vertical lines, use the x = img_width line directly
                        
                        # Top boundary (y = 0)
                        if y2 != y1:  # Avoid division by zero for horizontal lines
                            t_top = -y1 / (y2 - y1)
                            top_point = parametric_line(x1, y1, x2, y2, t_top)
                        else:
                            top_point = (x1, 0)  # For horizontal lines, use the y = 0 line directly
                        
                        # Bottom boundary (y = img_height)
                        if y2 != y1:  # Avoid division by zero for horizontal lines
                            t_bottom = (img_height - y1) / (y2 - y1)
                            bottom_point = parametric_line(x1, y1, x2, y2, t_bottom)
                        else:
                            bottom_point = (x1, img_height)  # For horizontal lines, use the y = img_height line directly

                        # Return the extended points
                        return (left_point, right_point, top_point, bottom_point)



 def parametric_line(x1, y1, x2, y2, t):
                        """
                        Given two points (x1, y1) and (x2, y2), return the point at parameter t
                        on the line segment from (x1, y1) to (x2, y2).
                        """
                        x = x1 + t * (x2 - x1)
                        y = y1 + t * (y2 - y1)
                        return [int(x), int(y)]
                    
def line_intersection_parametric(p1, p2, p3, p4):
    """
    Find the intersection point of two lines using parametric representation.
    Each line is defined by two points: (p1, p2) for the first line and (p3, p4) for the second.
    Returns the intersection point (x, y) if lines are not parallel, None otherwise.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    # Calculate the denominator
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    # If denominator is 0, lines are parallel, so no intersection
    if denom == 0:
        return None
    
    # Calculate the intersection point
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    s = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom
    
    # If t and s are between 0 and 1, the intersection is within the segment
    if 0 <= t <= 1 and 0 <= s <= 1:
        intersection = parametric_line(x1, y1, x2, y2, t)
        return intersection
    else:
        return None  # Intersection is outside the segments