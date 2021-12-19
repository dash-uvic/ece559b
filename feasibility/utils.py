import numpy as np
import torchvision.transforms as T

mean  = [0.485, 0.456, 0.406]
std   = [0.229, 0.224, 0.225]
normalize = T.Normalize(mean, std)
inv_normalize = T.Normalize(
        mean=[-mean[0]/std[0], -mean[1]/std[1], -mean[2]/std[2]],
        std=[1/std[0], 1/std[1], 1/std[2]]
    )

def iou_score(bbox,gtbox):
    bbox = np.float32(bbox)
    gtbox = np.float32(gtbox)

    yA = max(bbox[0], gtbox[0])
    xA = max(bbox[1], gtbox[1])
    yB = min(bbox[2], gtbox[2])
    xB = min(bbox[3], gtbox[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    bboxArea = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
    gtboxArea = (gtbox[2] - gtbox[0] + 1) * (gtbox[3] - gtbox[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(bboxArea + gtboxArea - interArea)

    # return the intersection over union value
    return iou
