import numpy as np
import cv2
from numpy.lib.stride_tricks import sliding_window_view
   

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image, save_dog_img=False):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = []

        base_img = image
        for octave_idx in range(self.num_octaves):
            if octave_idx != 0:
                base_img = gaussian_images[-1][-1]
                # Do downsampling
                height = base_img.shape[0] // 2
                width = base_img.shape[1] // 2
                base_img = cv2.resize(base_img, (width,height), interpolation=cv2.INTER_NEAREST)

            temp_imgs = [base_img]
            for step in range(1, self.num_DoG_images_per_octave + 1):
                temp_imgs.append(cv2.GaussianBlur(base_img, (0,0), self.sigma**step))
            gaussian_images.append(temp_imgs)
        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        for imgs in gaussian_images:
            temp_imgs = []
            for img_idx in range(1, self.num_guassian_images_per_octave):
                temp_imgs.append(cv2.subtract(imgs[img_idx], imgs[img_idx - 1]))
            dog_images.append(temp_imgs)

        # save dog images
        if save_dog_img:
            def normalize(img):
                # Do normalize
                return (img - img.min()) / (img.max() - img.min()) * 255
            for octave_idx in range(self.num_octaves):
                for step in range(self.num_DoG_images_per_octave):
                    cv2.imwrite(f"DoG{octave_idx+1}-{step+1}.png", normalize(dog_images[octave_idx][step]))

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        max_count = 0
        min_count = 0
        keypoints = []
        for octave_idx, imgs in enumerate(dog_images):
            windows_points = []
            # use sliding windows to get multiple windows from dog images
            for img in imgs:
                windows = sliding_window_view(img, window_shape=(3,3)).reshape(-1, 3*3)
                # window[4] is middle point
                windows_points.append([[window[4], window.max(), window.min()] for window in windows])

            # check local extremum for cube in three dog images
            width = imgs[0].shape[1] - 2
            for img_idx in range(1,len(windows_points) - 1):
                for window_idx, mid_window in enumerate(windows_points[img_idx]):
                    mid_v = mid_window[0]
                    max_v = max([windows_points[i][window_idx][1] for i in range(img_idx - 1, img_idx + 2)])
                    min_v = min([windows_points[i][window_idx][2] for i in range(img_idx - 1, img_idx + 2)])
                    if (mid_v <= min_v or mid_v >= max_v) and abs(mid_v) > self.threshold:
                        if mid_v <= min_v:
                            min_count += 1
                        else:
                            max_count += 1
                        row = ((window_idx // width) + 1) *(octave_idx + 1)
                        col = ((window_idx % width) + 1) *(octave_idx + 1)
                        keypoints.append([row, col])
        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(np.array(keypoints), axis=0)
        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))]
        return keypoints
