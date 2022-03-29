# https://github.com/dvlab-research/Parametric-Contrastive-Learning/blob/main/LT/losses.py
  
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .registry import register

class PrototypicalMOCO(nn.Module):
    def __init__(self, alpha):
        self.alpha
    
    def forward(self, centroids, labels, density):
        proto_labels = []
        proto_logits = []
        for n, (im2cluster, prototypes, density) in enumerate(zip(labels,centroids,density)):
            # get positive prototypes
            pos_proto_id = im2cluster[index]
            pos_prototypes = prototypes[pos_proto_id]    
            
            # sample negative prototypes
            all_proto_id = [i for i in range(im2cluster.max())]       
            neg_proto_id = set(all_proto_id)-set(pos_proto_id.tolist())
            neg_proto_id = sample(neg_proto_id,self.r) #sample r negative prototypes 
            neg_prototypes = prototypes[neg_proto_id]    

            proto_selected = torch.cat([pos_prototypes,neg_prototypes],dim=0)
            
            # compute prototypical logits
            logits_proto = torch.mm(q,proto_selected.t())
            
            # targets for prototype assignment
            labels_proto = torch.linspace(0, q.size(0)-1, steps=q.size(0)).long().cuda()
            
            # scaling temperatures for the selected prototypes
            temp_proto = density[torch.cat([pos_proto_id,torch.LongTensor(neg_proto_id).cuda()],dim=0)]  
            logits_proto /= temp_proto
            
            proto_labels.append(labels_proto)
            proto_logits.append(logits_proto)

@register("paco")
class PaCoLoss(nn.Module):
    def __init__(self, alpha, beta=1.0, gamma=1.0, supt=1.0, temperature=1.0, base_temperature=None, K=128, num_classes=1000):
        super(PaCoLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.supt = supt
        self.num_classes = num_classes

    def cal_weight_for_classes(self, cls_num_list):
        cls_num_list = torch.Tensor(cls_num_list).view(1, self.num_classes)
        self.weight = cls_num_list / cls_num_list.sum()
        self.weight = self.weight.to(torch.device('cuda'))

    def forward(self, features, labels=None, sup_logits=None, mask=None, epoch=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        ss = features.shape[0]
        batch_size = ( features.shape[0] - self.K ) // 2

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels[:batch_size], labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature)

        # add supervised logits
        anchor_dot_contrast = torch.cat(( (sup_logits + torch.log(self.weight + 1e-9) ) / self.supt, anchor_dot_contrast), dim=1)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # add ground truth 
        one_hot_label = torch.nn.functional.one_hot(labels[:batch_size,].view(-1,), num_classes=self.num_classes).to(torch.float32)
        mask = torch.cat((one_hot_label * self.beta, mask * self.alpha), dim=1)

        # compute log_prob
        logits_mask = torch.cat((torch.ones(batch_size, self.num_classes).to(device), self.gamma * logits_mask), dim=1)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
       
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss