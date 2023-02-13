import cv2
import numpy as np
import open3d as o3d

# numpy转点云格式
# points: n*3
def np_to_pcd(points, normals=None, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

# 相机投影过程
# points: n*3
# return: n*2
def project_points(points_3d, camera_intric):
    z = points_3d.T[2, :]
    xy1 = points_3d.T/z
    uv1 = camera_intric @ xy1
    return uv1[0:2].T