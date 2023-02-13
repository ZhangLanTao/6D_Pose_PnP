import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation
from params import camera_mtx
import numpy as np
import utils
import copy
import matplotlib.pyplot as plt
import time


def solve_pnp_ransac(pts_3d, pts_2d, camera_intric, initial_pose_guess=None):
    if initial_pose_guess is None:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(pts_3d, pts_2d, camera_intric, None, flags=cv2.SOLVEPNP_AP3P)
        rvec, tvec = cv2.solvePnPRefineLM(pts_3d[inliers], pts_2d[inliers], camera_intric, None, rvec, tvec)
        R = Rotation.from_rotvec(rvec.ravel())
    else:
        rvec, tvec = initial_pose_guess
        success, rvec, tvec, inliers = cv2.solvePnPRansac(pts_3d, pts_2d, camera_intric, None, rvec, tvec, flags=cv2.SOLVEPNP_ITERATIVE)
        rvec, tvec = cv2.solvePnPRefineLM(pts_3d[inliers], pts_2d[inliers], camera_intric, None, rvec, tvec)
        R = Rotation.from_rotvec(rvec.ravel())

    return R, tvec

def test_pnp():
    # 虚拟相机分辨率 1680*1280， FOV 22*16.5°
    virtual_camera_mtx = np.array([ [4321.,  0.,      840.],
                                    [0.,     4321.,   640.],
                                    [0.,     0.,      1.]])

    model_pts_3d = np.array([[75., -250., -100.],
                             [-75., -250., -100.],
                             [0., -280., -130.],
                             [0., -220., -130.]])  # n*3
    model_pc = utils.np_to_pcd(model_pts_3d)
    model_pc.paint_uniform_color([0, 1, 0])

    rotation_gt = Rotation.from_euler('xyz', [0.5,0.,0.])
    translation_gt = np.array([0.,300.,780.])

    ## diy模拟投影过程
    scene_pc = copy.deepcopy(model_pc)
    scene_pc.rotate(rotation_gt.as_matrix(), [0,0,0])
    scene_pc.translate(translation_gt)
    scene_pts_3d = np.asarray(scene_pc.points)
    pts_2d_diy = utils.project_points(scene_pts_3d, virtual_camera_mtx)

    ## opencv 模拟投影过程
    pts_2d = cv2.projectPoints(np.expand_dims(model_pts_3d,axis=1),
                               rotation_gt.as_matrix(), translation_gt,
                               virtual_camera_mtx, None)[0][:,0,:]
    # pts_2d = utils.project_points(scene_pts_3d, virtual_camera_mtx)
    noise = np.random.normal([0,0], [0.2, 0.2], pts_2d.shape)
    pts_2d = pts_2d+noise
    rotation, t = solve_pnp_ransac(model_pts_3d, pts_2d, virtual_camera_mtx)

    print("真值(mrad)", np.around(rotation_gt.as_euler('xyz')*1000, decimals=2))
    print("P3P欧拉角(mrad)", np.around(rotation.as_euler('xyz')*1000, decimals=2))
    print("角度误差(mrad)", np.around(rotation_gt.as_euler('xyz')*1000-rotation.as_euler('xyz')*1000, decimals=2))
    print("tvec 误差(mm)", np.around(translation_gt-t, decimals=2))

    # print()
    error = rotation_gt.as_euler('xyz')*1000-rotation.as_euler('xyz')*1000
    result_pc = copy.deepcopy(model_pc)
    result_pc = result_pc.rotate(rotation.as_matrix())
    result_pc = result_pc.translate(t.ravel())
    result_pc.paint_uniform_color([1,0,0])
    result_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(100)
    result_frame.rotate(rotation.as_matrix()).translate(t.ravel())


    # result_points = np.asarray(result_pc.points)
    # print("重投影误差", np.linalg.norm(pts_2d-utils.project_points(result_points, virtual_camera_mtx), axis=1))
    # o3d.visualization.draw_geometries([scene_pc, result_pc, result_frame, o3d.geometry.TriangleMesh.create_coordinate_frame(100)])
    # print()
    return error


if __name__ == '__main__':
    errors = []
    tic = time.time()
    for i in range(500):
        error = test_pnp()
        errors.append(error)
        plt.scatter(i, error[0])
    errors = np.array(errors)
    print("xyz轴标准差",np.var(errors[:,0]), np.var(errors[:,1]), np.var(errors[:,2]))
    toc = time.time()
    print("每帧pnp用时(ms）：", (toc-tic)*1000/500)

    plt.show()
