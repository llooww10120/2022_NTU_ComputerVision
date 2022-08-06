import numpy as np
import cv2.ximgproc as xip

def dist(a, b):
    # #L2, with half = 0
    # return np.sum(np.square(a - b), axis=-1)
 
    #L1, with half = 0
    return np.sum(np.abs(a - b), axis=-1)

    

def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both "Il to Ir" and "Ir to Il" for later left-right consistency
    Il_Ir_cost = np.zeros((max_disp+1, h, w), dtype=np.float32)
    Ir_Il_cost = np.zeros((max_disp+1, h, w), dtype=np.float32)
    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    for disp in range(max_disp + 1):
        Il_Ir_cost[disp, :, disp:] = dist(Il[:, disp: , ...], Ir[:, :w-disp, ...])
        Ir_Il_cost[disp, :, :w - disp] = dist(Ir[:, :w-disp , ...], Il[:, disp:, ...])
        Il_Ir_cost[disp, :, :disp] = np.expand_dims(Il_Ir_cost[disp, :, disp], axis=-1)
        Ir_Il_cost[disp, :, w - disp:] = np.expand_dims(Ir_Il_cost[disp, :, w - disp - 1], axis=-1)
        Il_Ir_cost[disp,] = xip.jointBilateralFilter(Il, Il_Ir_cost[disp,], 30, 5, 5)
        Ir_Il_cost[disp,] = xip.jointBilateralFilter(Ir, Ir_Il_cost[disp,], 30, 5, 5) 


    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    wta_disp_L = np.argmin(Il_Ir_cost, axis=0)
    wta_disp_R = np.argmin(Ir_Il_cost, axis=0)
    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    for y in range(h):
        for x in range(w):
            aaa = x-wta_disp_L[y,x]
            if aaa>=0 and wta_disp_L[y,x] == wta_disp_R[y,aaa]:
                continue
            else:
                wta_disp_L[y,x]=-1

    for y in range(h):
        for x in range(w):
            if wta_disp_L[y,x] == -1:
                l = 0
                r = 0
                while x-l>=0 and wta_disp_L[y,x-l] == -1:
                    l+=1
                if x-l < 0:
                    FL = max_disp 
                else:
                    FL = wta_disp_L[y,x-l]

                while x+r<=w-1 and wta_disp_L[y,x+r] == -1:
                    r+=1
                if x+r > w-1:
                    FR = max_disp
                else:
                    FR = wta_disp_L[y, x+r]
                wta_disp_L[y,x] = min(FL, FR)
    Il = Il.astype(np.uint8)
    wta_disp_L = wta_disp_L.astype(np.uint8)
    labels = xip.weightedMedianFilter(Il, wta_disp_L, 18, 1)

    return labels.astype(np.uint8)
    