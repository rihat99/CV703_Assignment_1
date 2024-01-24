from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights

from models.convnextv1 import *
from models.convnextv2 import *

from torch import nn
import torch
import torch.nn.functional as F


def get_resnet50(pretrained=False, num_classes=10, freeze=False):
    if pretrained:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Freeze model weights
        if freeze:
            for param in model.parameters():
                param.requires_grad = False

            for param in model.layer4.parameters():
                param.requires_grad = True

            for i, (name, layer) in enumerate(model.layer4.named_modules()):
                if isinstance(layer, torch.nn.Conv2d):
                    layer.reset_parameters()

    else:
        model = resnet50()
    

    # Add on fully connected layers for the output of our model

    # model.avgpool = torch.nn.Identity()

    model.fc = torch.nn.Sequential(
        # torch.nn.Dropout(p=0.2),
        torch.nn.Linear(
            in_features=2048,
            out_features=num_classes,
            bias=True
        )
    )
    
    return model


    

def get_model(model_name, pretrained=False, num_classes=10, freeze=False):
    if model_name == "ResNet50":
        return get_resnet50(pretrained, num_classes, freeze)
    elif model_name == "ConvNeXtTiny":
        return get_convnext_tiny(pretrained, num_classes, freeze)
    elif model_name == "ConvNeXtBase":
        return get_convnext_base(pretrained, num_classes, freeze)
    elif model_name == "ConvNeXtLarge":
        return get_convnext_large(pretrained, num_classes, freeze)
    elif model_name == "ConvTransNeXtLarge":
        return ConvTransNeXtLarge(pretrained, num_classes, freeze)
    elif model_name == "ConvTransNeXtTiny":
        return ConvTransNeXtTiny(pretrained, num_classes, freeze)
    elif model_name == "ConvTransNeXtBase":
        return ConvTransNeXtBase(pretrained, num_classes, freeze)
    
    elif model_name == "ConvNeXtV2Large":
        return convnextv2_large(pretrained, num_classes, freeze)

    else:
        print(model_name)
        raise Exception("Model not implemented")