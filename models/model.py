from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights

from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights

from torchvision.models import resnet101
from torchvision.models import ResNet101_Weights


from torchvision.models import convnext_tiny
from torchvision.models import ConvNeXt_Tiny_Weights

from torchvision.models import convnext_base
from torchvision.models import ConvNeXt_Base_Weights


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


def get_resnet18(pretrained=False, num_classes=10):
    if pretrained:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Freeze model weights
        for param in model.parameters():
            param.requires_grad = False

        for param in model.layer4.parameters():
            param.requires_grad = True

        for i, (name, layer) in enumerate(model.layer4.named_modules()):
            if isinstance(layer, torch.nn.Conv2d):
                layer.reset_parameters()

    else:
        model = resnet18()

    # Add on fully connected layers for the output of our model

    model.fc = torch.nn.Sequential(
        nn.Dropout(0.2),
        torch.nn.Linear(
            in_features=512,
            out_features=256,
            bias=True
        ),
        nn.ReLU(),
        nn.Dropout(0.2),
        torch.nn.Linear(
            in_features=256,
            out_features=num_classes,
            bias=True
        )
    )
    
    return model

def get_resnet101(pretrained=False):
    if pretrained:
        model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)

        # Freeze model weights
        for param in model.parameters():
            param.requires_grad = False

        for param in model.layer4.parameters():
            param.requires_grad = True

        for i, (name, layer) in enumerate(model.layer4.named_modules()):
            if isinstance(layer, torch.nn.Conv2d):
                layer.reset_parameters()

    else:
        model = resnet101()
    

    # Add on fully connected layers for the output of our model

    # model.avgpool = torch.nn.Identity()

    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(
            in_features=2048,
            out_features=68 * 2,
            bias=True
        )
    )
    
    return model

def _is_contiguous(tensor: torch.Tensor) -> bool:
    # jit is oh so lovely :/
    # if torch.jit.is_tracing():
    #     return True
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    else:
        return tensor.is_contiguous(memory_format=torch.contiguous_format)

class LayerNorm2d(nn.LayerNorm):
    r""" LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        if _is_contiguous(x):
            return F.layer_norm(
                x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
        else:
            s, u = torch.var_mean(x, dim=1, keepdim=True)
            x = (x - u) * torch.rsqrt(s + self.eps)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
            return x
        

def get_convnext_tiny(pretrained=False, num_classes=10, freeze=False):
    if pretrained:
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

        # Freeze model weights
        if freeze:
            for i in range(6):
                for param in model.features[i].parameters():
                    param.requires_grad = False

        for i in range(6, 8):
            for name, layer in model.features[i].named_modules():
                if isinstance(layer, torch.nn.Conv2d):
                    # print(i, name, layer)
                    layer.reset_parameters()

    else:
        model = convnext_tiny()



    model.classifier = torch.nn.Sequential(
        LayerNorm2d(768, eps=1e-06),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(
            in_features=768,
            out_features=num_classes,
            bias=True
        )
    )

    return model

def get_convnext_base(pretrained=False, num_classes=10, freeze=False):
    if pretrained:
        model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)

        # Freeze model weights
        if freeze:
            for i in range(6):
                for param in model.features[i].parameters():
                    param.requires_grad = False

        for i in range(6, 8):
            for name, layer in model.features[i].named_modules():
                if isinstance(layer, torch.nn.Conv2d):
                    # print(i, name, layer)
                    layer.reset_parameters()

    else:
        model = convnext_base()



    model.classifier = torch.nn.Sequential(
        LayerNorm2d(1024, eps=1e-06),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(
            in_features=1024,
            out_features=num_classes,
            bias=True
        )
    )

    return model
    

def get_model(model_name, pretrained=False, num_classes=10, freeze=False):
    if model_name == "ResNet50":
        return get_resnet50(pretrained, num_classes, freeze)
    elif model_name == "ResNet18":
        return get_resnet18(pretrained, num_classes)
    elif model_name == "ResNet101":
        return get_resnet101(pretrained)
    elif model_name == "ConvNeXtTiny":
        return get_convnext_tiny(pretrained, num_classes, freeze)
    elif model_name == "ConvNeXtBase":
        return get_convnext_base(pretrained, num_classes, freeze)
    else:
        raise Exception("Model not implemented")