import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES

@LOSSES.register_module()
class MyDiceLoss(nn.Module):
    def __init__(self, loss_weight=1.0, class_weight=None, smooth=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.smooth = smooth
        self.class_weight = class_weight
        print(f"MyDiceLoss initialized with weight={loss_weight}, class_weight={class_weight}")

    def forward(self, pred, target, ignore_index=255, **kwargs):
        print(f"MyDiceLoss forward called with pred={pred.shape}, target={target.shape}")
        
        # Convert pred to probabilities
        pred = F.softmax(pred, dim=1)
        
        # Handle target shape
        if pred.shape[2:] != target.shape[1:]:
            target = F.interpolate(
                target.unsqueeze(1).float(),
                size=pred.shape[2:],
                mode='nearest'
            ).squeeze(1).long()
        
        # Create mask
        mask = (target != ignore_index)
        target = target * mask
        
        # One-hot encode
        target_one_hot = F.one_hot(
            torch.clamp(target, 0, pred.shape[1]-1),
            num_classes=pred.shape[1]
        ).permute(0, 3, 1, 2).float()
        
        # Compute dice
        intersection = torch.sum(pred * target_one_hot, dim=(2, 3))
        union = torch.sum(pred + target_one_hot, dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice
        
        # Apply class weights
        if self.class_weight is not None:
            loss = loss * torch.tensor(self.class_weight, device=loss.device)
            
        return torch.mean(loss) * self.loss_weight