import numpy as np


class IoUMetric:
    def __init__(self, num_classes=19):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        
    def update(self, pred, target):
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        
        # Flatten the arrays
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        
        # Accumulate confusion matrix
        mask = (target >= 0) & (target < self.num_classes)
        if mask.sum() > 0:
            self.confusion_matrix += np.bincount(
                self.num_classes * target[mask].astype(int) + pred[mask],
                minlength=self.num_classes ** 2
            ).reshape(self.num_classes, self.num_classes)
    
    def compute(self):
        # Calculate IoU for each class
        intersection = np.diag(self.confusion_matrix)
        union = (self.confusion_matrix.sum(axis=1) + 
                self.confusion_matrix.sum(axis=0) - intersection)
        iou = intersection / (union + 1e-8)
        
        # Calculate mean IoU
        valid_classes = union > 0
        mean_iou = np.mean(iou[valid_classes])
        
        return mean_iou, iou
