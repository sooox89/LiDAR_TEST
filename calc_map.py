import numpy as np
import torch



def calc_iou():
    return 




"""
pred_boxes => [x,y,z,dx,dy,dz,yaw]
gt => 위와 동일

"""
def evaluate(pred_boxes, gt_boxes, scores, iou_thres):
        
    
    
        
    
    
    return ap, precision, recall








if __name__=="__main__":
    root = "/home/q/dataset/pandaset/demo_output"
    
    idx = 0
    
    pred_boxes = np.load(f"{root}/{idx:04d}_pred.npy")
    scores = np.load(f"{root}/{idx:04d}_score.npy")
    gt_boxes = np.load(f"{root}/{idx:04d}_gt.npy")
    
    iou_thresholds = [0.3, 0.5, 0.7]
    
    for iou_thres in iou_thresholds:
        ap, precision, recall = evaluate(pred_boxes, gt_boxes, scores, iou_thres = iou_thres)