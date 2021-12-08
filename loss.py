import torch
import torch.nn as nn

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()
    
    def forward(self, input, target):
        l2_loss = (target - input) ** 2
        l2_loss = torch.mean(l2_loss)

        return l2_loss

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
    
    def forward(self, input, target):
        
        return (torch.abs(input - target)).mean()
    

class MotionLoss(nn.Module):
    def __init__(self):
        super(MotionLoss, self).__init__()
        self.l2 = L2Loss()
        
    def forward(self, pred_mesh, pred_mesh_nxt, gt_mesh, gt_mesh_nxt):
        loss = 2 * self.l2((pred_mesh_nxt - pred_mesh), (gt_mesh_nxt - gt_mesh))
        
        return loss