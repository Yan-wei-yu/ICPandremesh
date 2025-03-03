# plybb.py
import numpy as np
from plyfile import PlyData

def get_bounding_box(vertices):
    return (np.min(vertices[:, 0]), np.max(vertices[:, 0]),
            np.min(vertices[:, 1]), np.max(vertices[:, 1]),
            np.min(vertices[:, 2]), np.max(vertices[:, 2]))

def read_ply(file_path):
    ply_data = PlyData.read(file_path)
    vertices = np.vstack([ply_data['vertex']['x'],
                          ply_data['vertex']['y'],
                          ply_data['vertex']['z']]).T
    return vertices

def get_depth_from_gray_value(gray_value, depth_max, depth_min, min_z, max_z):
    depth_scale = (depth_max - depth_min) / (max_z - min_z)
    z = (gray_value / depth_scale) + min_z
    return z

def get_centroid(vertices):
    return np.mean(vertices, axis=0)


def get_step_depth(gray_value, max_gray, min_gray, min_z, max_z, steps=3):
    """
    根據灰階值分段重建深度，適用於樓梯結構。
    """
    step_size = (max_gray - min_gray) / steps
    depth_step = (max_z - min_z) / steps
    # 將灰階分成多個深度階段
    step_index = int((gray_value - min_gray) // step_size)
    step_index = max(0, min(step_index, steps - 1))  # 防止超出範圍
    return min_z + step_index * depth_step


