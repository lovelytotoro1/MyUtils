import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet import *


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = resnet50()

        # the AdjustNeck network used for search and template respectively
        self.neck4s = AdjustNeck()
        self.neck4t = AdjustNeck()

        # the regression head
        self.head = Head()

    def forward(self, data):
        template = data['template'].cuda()
        search = data['search'].cuda()
        """
            获取其他各项target
        """

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        
        zf = self.neck4t(zf)
        xf = self.neck4s(xf)
        
        outputs = self.head(zf, xf)
        
        # get loss
#         cls = self.log_softmax(cls)
#         cls_loss = select_cross_entropy_loss(cls, label_cls)
#         loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        return outputs
