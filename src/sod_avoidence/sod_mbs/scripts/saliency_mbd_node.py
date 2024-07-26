#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import math
import scipy.spatial.distance
import skimage.color
from skimage.util import img_as_float

def raster_scan(img, L, U, D):
    n_rows = len(img)
    n_cols = len(img[0])

    for x in range(1, n_rows - 1):
        for y in range(1, n_cols - 1):
            ix = img[x][y]
            d = D[x][y]

            u1 = U[x-1][y]
            l1 = L[x-1][y]

            u2 = U[x][y-1]
            l2 = L[x][y-1]

            b1 = max(u1, ix) - min(l1, ix)
            b2 = max(u2, ix) - min(l2, ix)

            if d <= b1 and d <= b2:
                continue
            elif b1 < d and b1 <= b2:
                D[x][y] = b1
                U[x][y] = max(u1, ix)
                L[x][y] = min(l1, ix)
            else:
                D[x][y] = b2
                U[x][y] = max(u2, ix)
                L[x][y] = min(l2, ix)

    return True

def raster_scan_inv(img, L, U, D):
    n_rows = len(img)
    n_cols = len(img[0])

    for x in range(n_rows - 2, 1, -1):
        for y in range(n_cols - 2, 1, -1):
            ix = img[x][y]
            d = D[x][y]

            u1 = U[x+1][y]
            l1 = L[x+1][y]

            u2 = U[x][y+1]
            l2 = L[x][y+1]

            b1 = max(u1, ix) - min(l1, ix)
            b2 = max(u2, ix) - min(l2, ix)

            if d <= b1 and d <= b2:
                continue
            elif b1 < d and b1 <= b2:
                D[x][y] = b1
                U[x][y] = max(u1, ix)
                L[x][y] = min(l1, ix)
            else:
                D[x][y] = b2
                U[x][y] = max(u2, ix)
                L[x][y] = min(l2, ix)

    return True

def mbd(img, num_iters):
    if len(img.shape) != 2:
        print('did not get 2d np array to fast mbd')
        return None
    if (img.shape[0] <= 3 or img.shape[1] <= 3):
        print('image is too small')
        return None

    L = np.copy(img)
    U = np.copy(img)
    D = float('Inf') * np.ones(img.shape)
    D[0,:] = 0
    D[-1,:] = 0
    D[:,0] = 0
    D[:,-1] = 0

    img_list = img.tolist()
    L_list = L.tolist()
    U_list = U.tolist()
    D_list = D.tolist()

    for x in range(0, num_iters):
        if x % 2 == 1:
            raster_scan(img_list, L_list, U_list, D_list)
        else:
            raster_scan_inv(img_list, L_list, U_list, D_list)

    return np.array(D_list)

def get_saliency_mbd(img, method='b'):
    if img.shape[2] == 4: 
        img = img[:, :, :3]

    img_mean = np.mean(img, axis=(2))
    sal = mbd(img_mean, 3)

    if method == 'b':
        (n_rows, n_cols, n_channels) = img.shape
        img_size = math.sqrt(n_rows * n_cols)
        border_thickness = int(math.floor(0.1 * img_size))

        img_lab = img_as_float(skimage.color.rgb2lab(img))
        
        px_left = img_lab[0:border_thickness, :, :]
        px_right = img_lab[n_rows - border_thickness:, :, :]

        px_top = img_lab[:, 0:border_thickness, :]
        px_bottom = img_lab[:, n_cols - border_thickness:, :]
        
        px_mean_left = np.mean(px_left, axis=(0,1))
        px_mean_right = np.mean(px_right, axis=(0,1))
        px_mean_top = np.mean(px_top, axis=(0,1))
        px_mean_bottom = np.mean(px_bottom, axis=(0,1))

        px_left = px_left.reshape((n_cols*border_thickness, 3))
        px_right = px_right.reshape((n_cols*border_thickness, 3))

        px_top = px_top.reshape((n_rows*border_thickness, 3))
        px_bottom = px_bottom.reshape((n_rows*border_thickness, 3))

        def safe_invert(cov):
            try:
                return np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                # If inversion fails, use pseudo-inverse
                return np.linalg.pinv(cov)

        cov_left = safe_invert(np.cov(px_left.T))
        cov_right = safe_invert(np.cov(px_right.T))
        cov_top = safe_invert(np.cov(px_top.T))
        cov_bottom = safe_invert(np.cov(px_bottom.T))

        img_lab_unrolled = img_lab.reshape(img_lab.shape[0]*img_lab.shape[1], 3)

        def compute_distance(unrolled, mean, cov):
            mean_2 = np.zeros((1,3))
            mean_2[0,:] = mean
            dist = scipy.spatial.distance.cdist(unrolled, mean_2, 'mahalanobis', VI=cov)
            return dist.reshape((img_lab.shape[0], img_lab.shape[1]))

        u_left = compute_distance(img_lab_unrolled, px_mean_left, cov_left)
        u_right = compute_distance(img_lab_unrolled, px_mean_right, cov_right)
        u_top = compute_distance(img_lab_unrolled, px_mean_top, cov_top)
        u_bottom = compute_distance(img_lab_unrolled, px_mean_bottom, cov_bottom)

        # Normalize distances
        def normalize(u):
            max_u = np.max(u)
            return u / max_u if max_u != 0 else u

        u_left = normalize(u_left)
        u_right = normalize(u_right)
        u_top = normalize(u_top)
        u_bottom = normalize(u_bottom)

        u_max = np.maximum.reduce([u_left, u_right, u_top, u_bottom])

        u_final = (u_left + u_right + u_top + u_bottom) - u_max

        u_max_final = np.max(u_final)
        sal_max = np.max(sal)
        sal = sal / sal_max + u_final / u_max_final if u_max_final != 0 else sal / sal_max

    # postprocessing
    sal = sal / np.max(sal)
    
    s = np.mean(sal)
    alpha = 50.0
    delta = alpha * math.sqrt(s)

    xv, yv = np.meshgrid(np.arange(sal.shape[1]), np.arange(sal.shape[0]))
    (w, h) = sal.shape
    w2 = w/2.0
    h2 = h/2.0

    C = 1 - np.sqrt(np.power(xv - h2, 2) + np.power(yv - w2, 2)) / math.sqrt(np.power(w2, 2) + np.power(h2, 2))

    sal = sal * C

    def f(x):
        b = 10
        return 1.0 / (1.0 + math.exp(-b*(x - 0.5)))

    fv = np.vectorize(f)

    sal = sal / np.max(sal)

    sal = fv(sal)

    return sal * 255.0

class SaliencyDetector:
    def __init__(self):
        rospy.init_node('saliency_detector', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/D435i_camera/color/image_raw", Image, self.callback)
        self.saliency_pub = rospy.Publisher("/saliency_map", Image, queue_size=10)
        
        # Create windows for displaying images
        cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Saliency Map", cv2.WINDOW_NORMAL)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        try:
            # Compute saliency map
            saliency_map = get_saliency_mbd(cv_image).astype('uint8')

            # Convert saliency map to ROS image message
            saliency_msg = self.bridge.cv2_to_imgmsg(saliency_map, "mono8")
            saliency_msg.header = data.header  # Copy the header from the input image
            self.saliency_pub.publish(saliency_msg)

            # Display original image and saliency map
            cv2.imshow("Original Image", cv_image)
            cv2.imshow("Saliency Map", saliency_map)
            cv2.waitKey(1)  # Wait for a millisecond to update the window

        except Exception as e:
            rospy.logerr(f"Error in saliency computation: {e}")

    def __del__(self):
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        sd = SaliencyDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass