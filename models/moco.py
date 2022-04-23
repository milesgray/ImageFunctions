import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import torchvision.models as models
from models import register
from models import create as create_model
from models.layers import create as create_layer
from models.layers import PixelAttention, SplitBatchNorm
from utility import make_coord
from einops import repeat

class ModelBase(nn.Module):
    """
    Creates a torchvision model and replaces some of the layers to 
    fit into the MOCO model.  
    """
    def __init__(self, feature_dim=128, arch=None, bn_splits=16, pa_layers=3):
        super().__init__()

        # use split batchnorm
        norm_layer = partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else nn.BatchNorm2d        
        resnet_arch = getattr(models, arch)
        net = resnet_arch(num_classes=feature_dim, norm_layer=norm_layer)

        pa_count = 0
        out_channels = 0
        self.net = []
        for name, module in net.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                out_channels = module.out_channels
            #elif isinstance(module, nn.ReLU):
                self.net.append(module)
                module = PixelAttention(out_channels)
            elif isinstance(module, nn.Sequential):
                if pa_count < pa_layers:
                    print(module._modules['1']._modules.keys())
                    out_size = module._modules['1']._modules['conv2'].out_channels
                    self.net.append(module)
                    module = PixelAttention(out_size, 
                                            channel_wise=pa_count%2==0,
                                            spatial_wise=pa_count%2!=0)
                    pa_count += 1
            elif isinstance(module, nn.MaxPool2d):
                continue
            elif isinstance(module, nn.Linear):
                self.net.append(nn.Flatten(1))
                
            self.net.append(module)

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        x = self.net(x)
        # note: not normalized here
        return x

@register('moco')
class ModelMoCo(nn.Module):
    def __init__(self, max_classes=500, dim=128, K=4096, m=0.99, 
                 arch='resnet18', bn_splits=8, pa_layers=3, 
                 symmetric=True, verbose=False, normalize=False,
                 margins={"supervised":0.1, "distance": 0.1, "centroid": 0.01},
                 temps={"contrastive": 0.1, "supervised": 0.07},
                 loss_mods={"contrastive": 1.0, "supervised": 0.1, "distance": 0.01, "imprint": 0.5}):                        
        super().__init__()
        self.init_log = True

        self.max_classes = max_classes
        self.K = K
        self.m = m
        self.T = temps["contrastive"]
        self.clr_t = temps["supervised"]
        self.clr_m = margins["supervised"]
        self.cen_m = margins["distance"]
        self.symmetric = symmetric
        self.verbose = verbose
        self.margins = margins
        self.temps = temps
        self.loss_mods = loss_mods

        # create the encoders
        self.encoder_q = ModelBase(feature_dim=dim, arch=arch, bn_splits=bn_splits, pa_layers=pa_layers)
        self.encoder_k = ModelBase(feature_dim=dim, arch=arch, bn_splits=bn_splits, pa_layers=pa_layers)
        self.linear = nn.Linear(dim, max_classes)
        self.linear_k = nn.Linear(dim, max_classes)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.linear.parameters(), self.linear_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("lbl_queue", torch.floor(torch.randn(1, K)).int())

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("centroids", torch.zeros(dim, self.max_classes))

        self.layer = -1
        self.feat_after_avg_k = None
        self.feat_after_avg_q = None
        #self._register_hook()

        self.normalize = normalize        

    def _find_layer(self, module):
        if type(self.layer) == str:
            modules = dict([*module.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*module.children()]

            return children[self.layer]
        return None

    def _hook_k(self, _, __, output):
        self.feat_after_avg_k = flatten(output)
        if self.normalize: 
           self.feat_after_avg_k = nn.functional.normalize(self.feat_after_avg_k, dim=1)


    def _hook_q(self, _, __, output):
        self.feat_after_avg_q = flatten(output)
        if self.normalize:
           self.feat_after_avg_q = nn.functional.normalize(self.feat_after_avg_q, dim=1)


    def _register_hook(self):
        layer_k = self._find_layer(self.encoder_k)
        assert layer_k is not None, f'hidden layer ({self.layer}) not found'
        handle = layer_k.register_forward_hook(self._hook_k)
        if self.init_log: print(f"[STARTUP]\thook for layer k registered")

        layer_q = self._find_layer(self.encoder_q)
        assert layer_q is not None, f'hidden layer ({self.layer}) not found'
        handle = layer_q.register_forward_hook(self._hook_q)
        if self.init_log: print(f"[STARTUP]\thook for layer q registered")


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.linear.parameters(), self.linear_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        if self.init_log: print(f"[STARTUP]\tEMA updated")

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels=None):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        if self.init_log: print(f"[STARTUP]\tkey queue updated:\n\tqueue:\t{self.queue[:, ptr:ptr + batch_size]}\n\tpointer:\t{ptr}")
        if self.init_log: print(f"[STARTUP]\tlabel queue updated:\n\tqueue:\t{self.lbl_queue[:, ptr:ptr + batch_size]}\n\tpointer:\t{ptr}")
        if labels is not None:
            self.lbl_queue[:, ptr:ptr + batch_size] = labels.t()
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr        

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def contrastive_loss(self, feat_q, feat_k):
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [feat_q, feat_k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [feat_q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.temps["contrastive"]

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        loss = nn.CrossEntropyLoss().cuda()(logits, labels)

        if self.init_log: print(f"[STARTUP]\t[contrastive loss]\t{loss}")

        return loss.mean()

    def supervised_contrastive_loss(self, feat_q, feat_k, labels, 
                                    density=torch.ones(args.batch_size),verbose=False):
        batch_size = feat_q.shape[0]
        dim = feat_q.shape[1]
        
        density = density.contiguous().view(-1, 1)
        labels = torch.cat([labels.contiguous().view(-1, 1), self.lbl_queue.clone().view(-1, 1)])
        pos_mask = torch.eq(labels, labels.T)
        pos_mask = pos_mask.float().to(feat_q.device)

        # positive logits: Nx1
        #l_pos = torch.div(torch.einsum('nc,nc->n', [feat_q, feat_k]).unsqueeze(-1),
                          #self.temps["supervised"])        
        l_pos = torch.div(
            torch.matmul(feat_q, feat_k.T),
            self.temps["supervised"]
        )
        m_pos, _ = torch.max(l_pos, dim=1, keepdim=True)
        pos_logits = torch.div(torch.exp(l_pos - m_pos.detach()),
                               density)
        # negative logits: NxK
        #l_neg = torch.div(torch.einsum('nc,ck->nk', [feat_q, self.queue.clone().detach()]),
        #                  self.temps["supervised"])
        l_neg = torch.div(
            torch.matmul(feat_q, self.queue.clone().detach()),
            self.temps["supervised"]
        )
        m_neg, _ = torch.max(l_neg, dim=1, keepdim=True)
        neg_logits = torch.exp(l_neg - m_neg.detach())
        neg_logits = torch.mul(neg_logits, density)
        
        if verbose: print(f"Feat q: {feat_q.shape}, feat_k: {feat_k.shape}, pos_logits: {pos_logits.shape}, neg_logits: {neg_logits.shape}")
        loss = torch.div(
            torch.log(
                torch.div(pos_logits.sum(1),
                          neg_logits.sum(1)).sum(-1)), 
            torch.squeeze(-1 * (2 * (pos_mask.sum(1) - 1))) - 1 # subtract 1 to avoid divide by 0 on classes with no examples this batch
        )

        if verbose: print(f"loss: {loss.shape}")
        if self.init_log: print(f"[STARTUP]\t[supervised contrastive loss]\t{loss}")
 
        return loss.mean()

    def spherical_dist_loss(self, feats, centroids, density=None):
        dist = feats - centroids
        dist = (dist.abs() - self.margins["distance"]) * dist.sign()
        loss = dist.norm(dim=-1).div(2).arcsin().pow(2).mul(2)

        if density is not None:
            loss /= density.contiguous()

        if self.init_log: print(f"[STARTUP]\t[spherical distance loss]\t{loss}")
        
        return loss.mean()

    def centroid_contrastive_loss(self, feat_q, feat_k, centroid, density, labels, 
                                  min=0, max=5):
        batch_size = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()
        centroid = centroid
        density = density.view(-1, 1)
        labels = labels.view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float().to(feat_q.device)
        scale = torch.squeeze(-1 * pos_mask.sum(-1)) - 1

        distance_scale = 100
        centroid_dist_a = euclidean_weighted_distance(feat_q, centroid, self.clr_t, density, self.cen_m, scale, 1/self.clr_t * distance_scale)
        #centroid_dist_b = euclidean_weighted_distance(feat_k, centroid, self.clr_t, density, self.cen_m, scale, 1/self.clr_t * distance_scale)
        loss = torch.clamp(centroid_dist_a * 0.5, min=min, max=max)# + torch.clamp(centroid_dist_b * 0.5, min=min, max=max)

        if self.init_log: print(f"[STARTUP]\t[centroid constrastive loss]\t{loss}")
        
        return self._clean(loss)

    def temporal_contrastive_loss(self, feat_q, feat_k, centroid, density, labels, ordering):
        ordered_index = torch.argsort(ordering)
        ordered_feat_q = feat_q[ordered_index]
        ordered_feat_k = feat_k[ordered_index].detach()
        ordered_centroid = centroid[ordered_index]

        loss = torch.nn.CTCLoss()
        if self.init_log: print(f"[STARTUP]\t[temporal contrastive loss]\t{loss}")
        return self._clean(loss)
    def encode_q(self, data):
        feat_q = self.encoder_q(data)  # queries: NxC
        feat_q = F.normalize(feat_q, dim=1)  # already normalized
        return feat_q

    def encode_k(self, data):
        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(data)

            feat_k = self.encoder_k(im_k_)  # keys: NxC
            feat_k = F.normalize(feat_k, dim=1)  # already normalized

            # undo shuffle
            feat_k = self._batch_unshuffle_single_gpu(feat_k, idx_unshuffle)

        return feat_k

    def forward(self, im1, im2, labels=None, centroids=None, density=None, verbose=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """

        losses = {}

        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        feat_q = self.encode_q(im1)
        #logits_q = self.linear(self.feat_after_avg_q)
        feat_k = self.encode_k(im2)    
        
        # compute loss
        if self.symmetric:  # symmetric loss
            if labels is not None:
                losses["supervised"] = self.supervised_contrastive_loss(feat_q, feat_k, labels, density=density, verbose=verbose)
                c_loss_12 = self.contrastive_loss(feat_q, feat_k)
                c_loss_21 = self.contrastive_loss(feat_k, feat_q)
                losses["contrastive"] = c_loss_12 + c_loss_21 
            else:
                s_loss = 0
                loss_12 = self.contrastive_loss(feat_q, feat_k)
                loss_21 = self.contrastive_loss(feat_k, feat_q)
                losses["contrastive"] = loss_12 + loss_21
            self._dequeue_and_enqueue(torch.cat([feat_k, feat_q], dim=0), 
                                      labels=torch.cat([labels, labels], dim=0))
        else:  # asymmetric loss
            if labels is not None:
                losses["supervised"] = self.supervised_contrastive_loss(feat_q, feat_k, labels, density=density, verbose=verbose)
                losses["contrastive"] = self.contrastive_loss(feat_q, feat_k)            
            self._dequeue_and_enqueue(feat_k, labels=labels)

        if centroids is not None:
            losses["dist"] = self.spherical_dist_loss(feat_q, centroids) * self.loss_mods["distance"]
        losses["imprint"] = nn.KLDivLoss(reduction="batchmean")(feat_q, centroids) * self.loss_mods["imprint"]
        if self.init_log: 
            print(f"[STARTUP]\tfinal loss ouput:\n\t\t{losses}")
            self.init_log = False
        return losses