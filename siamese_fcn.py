from collections import OrderedDict
import copy

import torchvision
import torch
from torch import nn
from torch.nn import functional as F

def siamese_fcn(backbone='resnet50'):
    """Constructs a Fully-Convolutional Network model with a ResNet backbone.
    Args:
        backbone: name of the backbone to be used

    """
    backbone = torchvision.models.resnet.__dict__[backbone](replace_stride_with_dilation=[False, True, True])

    return_layers = {'layer4': 'out'}
    backbone = torchvision.models._utils.IntermediateLayerGetter(backbone, return_layers=return_layers)

    inplanes = 2048
    model = SiameseFCN(backbone, torchvision.models.segmentation.fcn.FCNHead(inplanes, 1))
    
    return model

class SiameseFCN(nn.Module):

    def __init__(self, backbone, classifier):
        super(SiameseFCN, self).__init__()
        self.backbone1 = backbone
        self.backbone2 = copy.deepcopy(backbone)
        self.classifier = classifier

    def forward(self, im, targ):
        input_shape = im.shape[-2:]
        
        # contract: features is a dict of tensors
        features_im = self.backbone1(im)["out"]
        features_targ = self.backbone2(targ)["out"].view(features_im.shape[0], features_im.shape[1], -1, features_im.shape[-1])

        result = OrderedDict()
        x = torch.cat((features_im, features_targ), dim=2)
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        return result
