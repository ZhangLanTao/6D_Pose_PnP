import cv2
from scipy.spatial.transform import Rotation
from params import camera_mtx

def solve_pnp(pts_3d, pts_2d, initial_pose_guess):
    if initial_pose_guess is None:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(pts_3d, pts_2d, camera_mtx, None, flags=cv2.SOLVEPNP_ITERATIVE)
        rvec, tvec = cv2.solvePnPRefineLM(pts_3d[inliers], pts_2d[inliers], camera_mtx, None, rvec, tvec)
        R = Rotation.from_rotvec(rvec)
    else:
        rvec, tvec = initial_pose_guess
        success, rvec, tvec, inliers = cv2.solvePnPRansac(pts_3d, pts_2d, camera_mtx, None, rvec, tvec, flags=cv2.SOLVEPNP_ITERATIVE)
        rvec, tvec = cv2.solvePnPRefineLM(pts_3d[inliers], pts_2d[inliers], camera_mtx, None, rvec, tvec)
        R = Rotation.from_rotvec(rvec)

    return R, tvec