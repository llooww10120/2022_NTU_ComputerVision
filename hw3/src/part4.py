import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)

    wide = 0
    orb = cv2.ORB_create()
    bfmatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    # for all images to be stitched:
    for idx in range(len(imgs) - 1):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]
        h,w,_ = im2.shape
        wide  += w     
  
        # TODO: 1.feature detection & matching
        
        kp1, des1 = orb.detectAndCompute(im1, None)
        kp2, des2 = orb.detectAndCompute(im2, None)
        matches = bfmatcher.knnMatch(des1, des2, k=2)
 
        u = []
        v = []

        for m,n in matches:
            if m.distance < 0.7 * n.distance:
                u.append(kp1[m.queryIdx].pt)
                v.append(kp2[m.trainIdx].pt)
        u = np.array(u)
        v = np.array(v)

        # TODO: 2. apply RANSAC to choose best H
        num = 3000
        threshold = 4
        inlineNmax = 0
        best_H = np.zeros((3,3))
        indices = [i for i in range(len(u)-1)]
        for i in range(0, num+1):
            ran_u = np.zeros((4,2))
            ran_v = np.zeros((4,2))
            
            random.shuffle(indices)
            for j in range(4):
                ran_u[j] = u[indices[j]]
                ran_v[j] = v[indices[j]]
            H = solve_homography(ran_v, ran_u)
            
            con_v = np.concatenate((np.transpose(v), np.ones((1,len(v)))), axis=0)
            con_u = np.concatenate((np.transpose(u), np.ones((1,len(u)))), axis=0)             
            uv = np.dot(H,con_v)
            uv = uv/uv[-1,:]

            err  = np.linalg.norm((uv-con_u)[:-1,:], ord=1, axis=0)
            inlineN = sum(err<threshold) 
            if inlineN > inlineNmax:
                inlineNmax = inlineN
                best_H = H
        # TODO: 3. chain the homographies

        # TODO: 4. apply warping
        last_best_H = np.dot(last_best_H,best_H)
        out = warping(im2, dst, last_best_H, 0, h, wide, wide+w, direction='b') 
    return out

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)
