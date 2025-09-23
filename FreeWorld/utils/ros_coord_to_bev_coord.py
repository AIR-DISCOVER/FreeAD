import numpy as np
from pyquaternion import Quaternion
from FreeWorld.utils.coord_transformation import *

def ROSCoord2NuscenesBEVCoord(data):
    """
    Convert all BEV, map, and plan results from ROS(x, y) -> (-y, x) NuscenesBEV coordinate.
    Also converts the rotation quaternion for BEV and adjusts relevant points in map and plan data.
    """

    # Helper function to transform a quaternion for a 90-degree counterclockwise rotation
    def transform_quaternion(quat):
        q = Quaternion(quat)  # Create a quaternion object from the input
        rotate_90_deg = Quaternion(axis=[0, 0, 1], angle=-np.pi / 2)  # 90-degree clockwise rotation around Z (left-hand)
        new_quat = rotate_90_deg * q  # Apply the rotation
        return new_quat.elements  # Return the rotated quaternion

    # Process BEV results (handles box translation and rotation)
    if 'results' in data:
        for sample_token, prediction in data['results'].items():
            for obj in prediction:
                # Convert the box center from (x, y) to (-y, x)
                obj['translation'][0], obj['translation'][1] = -obj['translation'][1], obj['translation'][0]

                # Convert the rotation quaternion using the defined matrix
                if 'rotation' in obj:
                    q = Quaternion(obj['rotation'])  # Create a quaternion object from the rotation
                    # Apply a 90-degree rotation to the quaternion (clockwise around Z-axis)
                    rotate_90_deg = Quaternion(axis=[0, 0, 1], angle=-np.pi / 2)  # Clockwise 90 degrees for left-hand coord
                    new_rotation = rotate_90_deg * q
                    obj['rotation'] = new_rotation.elements  # Update the rotation with the new quaternion

    # Process map results (handles points defining boundaries or dividers)
    if 'map_results' in data:
        map_results = data['map_results']
        for sample_token, vectors in map_results.items():
            for vector in vectors['vectors']:
                if 'pts' in vector:
                    for pt in vector['pts']:
                        if len(pt) >= 2:
                            # Convert map points: from right-hand to left-hand system
                            pt[0], pt[1] = RightHandCoord2LeftHandCoord(pt[0], pt[1])  # Apply the coordinate transformation

    # Process plan results (handles nested tensors with planning points)
    if 'plan_results' in data:
        plan_results = data['plan_results']
        for sample_id, tensors in plan_results.items():
            for tensor in tensors:
                for points in tensor:
                    for point in points:
                        if len(point) >= 2:
                            # Convert plan points: from right-hand to left-hand system
                            point[0], point[1] = RightHandCoord2LeftHandCoord(point[0], point[1])  # Apply the coordinate transformation

    return data
