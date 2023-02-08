import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import blob_detector
import params
from solve_pnp import solve_pnp
from pose_filter import slide_window_filter
from params import camera_mtx

def show_result(img, pts_2d, R_filtered, t_filtered):
    cv2.drawFrameAxes(img, camera_mtx, None, R_filtered.as_rotvec(), t_filtered, length=10)
    cv2.imshow("result", img)

def main():
    camera = cv2.VideoCapture()
    bd = blob_detector.blob_detector()
    filter = slide_window_filter()
    initial_pose_guess = None
    while camera.isOpened():
        ret, frame = camera.read()
        frame_undistored = cv2.undistort(frame, params.camera_mtx, params.dist_coeffs)

        pts_2d = bd.detect(frame_undistored)

        R, t = solve_pnp(params.pts_3d, pts_2d, initial_pose_guess)

        R_filtered, t_filtered = filter.update(R, t)
        initial_pose_guess = [R_filtered, t_filtered]

        show_result(frame_undistored, pts_2d, R_filtered, t_filtered)

        key = cv2.waitKey(1)
        if key == 27:
            break
    print("退出")

if __name__ == '__main__':
    main()