import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    ux = u[:,0].reshape((N,1))
    uy = u[:,1].reshape((N,1))
    vx = v[:,0].reshape((N,1))
    vy = v[:,1].reshape((N,1))
    a = np.concatenate( (ux, uy, np.ones((N,1)), np.zeros((N,3)), -1*np.multiply(ux,vx), -1*np.multiply(uy,vx), -1*vx), axis=1 )
    aa = np.concatenate( (np.zeros((N,3)), ux, uy, np.ones((N,1)), -1*np.multiply(ux, vy), -1*np.multiply(uy,vy), -1*vy), axis=1 )
    A = np.concatenate((a, aa), axis=0)

    # TODO: 2.solve H with A
    _ , _ , VT = np.linalg.svd(A)
    H = VT[-1,:]/VT[-1,-1]
    H = H.reshape(3, 3)
    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    w = xmax-xmin
    h = ymax-ymin
    x, y = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))
    

    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    shape = (1,w*h)
    map = np.concatenate((x.reshape(shape), y.reshape(shape), np.ones(shape)), axis = 0)
    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        map = np.dot(H_inv,map)
        map = map/map[-1,:] 
        src_y = np.around(map[1,:].reshape((h, w))).astype(int)
        src_x = np.around(map[0,:].reshape((h, w))).astype(int)  
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        mask_h = (0 < src_y)*(src_y < h_src)
        mask_w = (0 < src_x)*(src_x < w_src)
        mask   = mask_h * mask_w
        
        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        # TODO: 6. assign to destination image with proper masking
        dst[y[mask], x[mask]] = src[src_y[mask], src_x[mask]]
        pass


    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        map = np.dot(H,map)
        map = map/map[-1,:]
        dst_y = np.around(map[1,:].reshape(h,w)).astype(int)
        dst_x = np.around(map[0,:].reshape(h,w)).astype(int)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        mask_h = (0 < dst_y)*(dst_y < h_dst)
        mask_w = (0 < dst_x)*(dst_x < w_dst)
        mask   = mask_h * mask_w
        # TODO: 5.filter the valid coordinates using previous obtained mask       
        # TODO: 6. assign to destination image using advanced array indicing
        dst[dst_y[mask], dst_x[mask]] = src[y[mask], x[mask]]

        pass

    return dst
