import numpy as np
from pyquaternion import Quaternion


def ROSCoord2NuscenesBEVCoord(data):
    """
    right hand coord to left hand coord
    
    Convert all BEV, map, and plan results from ROS(x, y) -> (-y, x) NuscenesBEV coordinate.
    Also converts the rotation quaternion for BEV and adjusts relevant points in map and plan data.
    """

    # Helper function to transform a point (x, y) -> (-y, x)
    def right_hand_coord_to_left_hand_coord(point):
        x = point[0]
        y = point[1]
        if len(point) < 2:
            raise ValueError(f"Point does not have enough dimensions for transformation: {point}")
        x, y = -y, x  # Modify in place

    # Helper function to transform a quaternion for a 90-degree counterclockwise rotation
    def transform_quaternion(quat):
        q = Quaternion(quat)  # Create a quaternion object from the input
        rotate_90_deg = Quaternion(axis=[0, 0, 1], angle=np.pi / 2)  # 90-degree CCW rotation around Z
        new_quat = rotate_90_deg * q  # Apply the rotation
        return new_quat.elements  # Return the rotated quaternion

    # Process BEV results (handles box translation and rotation)
    if 'results' in data:
        # Iterate over the BEV results to apply the coordinate transformation
        for sample_token, prediction in data['results'].items():
            for obj in prediction:
                # Convert the box center from (x, y) to (-y, x)
                obj['translation'][0], obj['translation'][1] = -obj['translation'][1], obj['translation'][0]
                
                # Convert the rotation quaternion using the defined matrix
                # Assuming the rotation quaternion is in obj['rotation']
                if 'rotation' in obj:
                    q = Quaternion(obj['rotation'])  # Create a quaternion object from the rotation
                    # Apply a 90-degree rotation to the quaternion
                    # First, construct the quaternion corresponding to the 90-degree counterclockwise rotation in 2D.
                    rotate_90_deg = Quaternion(axis=[0, 0, 1], angle=np.pi / 2)
                    # Multiply the current quaternion by the 90-degree quaternion (apply the transformation)
                    new_rotation = rotate_90_deg * q
                    obj['rotation'] = new_rotation.elements  # Update the rotation with the new quaternion
            

    # Process map results (handles points defining boundaries or dividers)
    if 'map_results' in data:
        map_results = data['map_results']
        for sample_token, vectors in map_results.items():
            for vector in vectors['vectors']:
                # Transform points defining boundaries or dividers
                if 'pts' in vector:
                    for pt in vector['pts']:
                        if len(pt) >= 2:
                            right_hand_coord_to_left_hand_coord(pt)  # Modify the point in place

    # Process plan results (handles nested tensors with planning points)
    if 'plan_results' in data:
        plan_results = data['plan_results']
        for sample_id, tensors in plan_results.items():
            for tensor in tensors:
                for points in tensor:
                    for point in points:
                        if len(point) >= 2:
                            right_hand_coord_to_left_hand_coord(point[:2])  # Only modify the first two dimensions in place

    return data
