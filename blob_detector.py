import timeit
from cv2 import cv2
import numpy as np

class blob_detector:
    def __init__(self):
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 80
        params.thresholdStep = 30
        params.maxThreshold = 255
    
        # Filter by color
        params.filterByColor = True
        params.blobColor = 255
    
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 15
        params.maxArea = 250
    
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.4
    
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.7
    
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.15
    
        # Create a detector with the parameters
        self.detector = cv2.SimpleBlobDetector_create(params)
    def detect(self, gray_img, debug=False):
        # Detect blobs.
        keypoints = self.detector.detect(gray_img)
        pts = np.array([keypoints[idx].pt for idx in range(0, len(keypoints))]).astype(np.float32)
        if debug:
            # Draw detected blobs as red circles.
            # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
            # the size of the circle corresponds to the size of blob
            im_with_keypoints = cv2.drawKeypoints(gray_img, keypoints, np.array([]), (0, 0, 255),
                                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
            # Show blobs
            print("共%s个关键点"%len(pts))
            cv2.imshow("Keypoints", cv2.resize(im_with_keypoints, (0,0), fx=0.3, fy=0.3))
            cv2.waitKey(0)
        return pts

if __name__ == '__main__':
    bd = blob_detector()
    img = cv2.imread("../data/20220823_20/L/plane3m/000000.png", cv2.IMREAD_GRAYSCALE)
    tic = timeit.default_timer()
    bd.detect(img, True)
    toc = timeit.default_timer()
    print("time:", toc-tic)
