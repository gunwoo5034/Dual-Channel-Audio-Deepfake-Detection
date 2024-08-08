import torch.nn.functional as F
import torch.nn as nn
import torch


class CustomLoss:
    def __init__(self,  weight_mse=1.0, weight_ce=1.0):
        self.weight_mse = weight_mse
        self.weight_ce = weight_ce

    def mse_loss(self, output, target):
        criterion=nn.MSELoss()
        return criterion(output, target) * self.weight_mse

    def ce_loss(self, output, target):
        criterion=nn.BCEWithLogitsLoss()
        
        return criterion(output, target) * self.weight_ce
    
    def calc_loss(self, img_out, aud_out, target, hyper_param=0.99):
        batch_size = target.size(0)
        loss = 0
        for batch in range(batch_size):
            dist = torch.dist(img_out[batch,:].view(-1), aud_out[batch,:].view(-1), 2)
            tar = target[batch,:].view(-1)
            loss += ((tar*(dist**2)) + ((1-tar)*(max(hyper_param-dist,0)**2)))
        return loss.mul_(1/batch_size)
    




