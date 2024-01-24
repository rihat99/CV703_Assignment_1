from torchvision.models import convnext_tiny
from torchvision.models import ConvNeXt_Tiny_Weights

from torchvision.models import convnext_base
from torchvision.models import ConvNeXt_Base_Weights

from torchvision.models import convnext_large
from torchvision.models import ConvNeXt_Large_Weights


from torch import nn
import torch
import torch.nn.functional as F

import math


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

def get_convnext_large(pretrained=False, num_classes=10, freeze=False):
    if pretrained:
        model = convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1)

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
        model = convnext_large()

    model.classifier = torch.nn.Sequential(
        LayerNorm2d(1536, eps=1e-06),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(
            in_features=1536,
            out_features=num_classes,
            bias=True
        )
    )

    return model

class ConvTransNeXtLarge(nn.Module):
    def __init__(self, pretrained=True, num_classes=200, freeze=True):
        super().__init__()
        
        if pretrained:
            self.conv_model = convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1)

            # Freeze model weights
            if freeze:
                for i in range(6):
                    for param in self.conv_model.features[i].parameters():
                        param.requires_grad = False

        else:
            self.conv_model = convnext_large()
            
        self.conv_model.features = nn.Sequential(*self.conv_model.features[:7])

        self.pos_encoder = PositionalEncoding(1536, dropout=0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=1536,  # This should match the feature size of ConvNeXt's last layer
            nhead=8,      # Number of attention heads
            dim_feedforward=1024,
            dropout=0.1,
            activation='gelu',
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.conv_model.classifier = torch.nn.Sequential(
            nn.Linear(
                in_features=1536,
                out_features=num_classes,
                bias=True
            )
    )

    def forward(self, x):
        x = self.conv_model.features(x)

        b, c, h, w = x.size()
        x = x.view(b, c, h * w).permute(2, 0, 1)

        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        x = torch.mean(x, dim=0)

        x = self.conv_model.classifier(x)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class ConvTransNeXtTiny(nn.Module):
    def __init__(self, pretrained=True, num_classes=200, freeze=True):
        super().__init__()
        
        if pretrained:
            self.conv_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

            # Freeze model weights
            if freeze:
                for i in range(6):
                    for param in self.conv_model.features[i].parameters():
                        param.requires_grad = False

        else:
            self.conv_model = convnext_tiny()
            
        self.conv_model.features = nn.Sequential(*self.conv_model.features[:6])

        self.pos_encoder = PositionalEncoding(384, dropout=0.2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=384,  # This should match the feature size of ConvNeXt's last layer
            nhead=8,      # Number of attention heads
            dim_feedforward=1024,
            dropout=0.2,
            activation='gelu',
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.conv_model.classifier = torch.nn.Sequential(
            nn.Linear(
                in_features=384,
                out_features=num_classes,
                bias=True
            )
    )

    def forward(self, x):
        x = self.conv_model.features(x)

        b, c, h, w = x.size()
        x = x.view(b, c, h * w).permute(2, 0, 1)

        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # x = x.permute(0, 2, 1).view(b, c, h, w)

        # x = self.conv_model.avgpool(x)

        x = torch.mean(x, dim=0)

        x = self.conv_model.classifier(x)

        return x

class ConvTransNeXtBase(nn.Module):
    def __init__(self, pretrained=True, num_classes=200, freeze=True):
        super().__init__()
        
        if pretrained:
            self.conv_model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)

            # Freeze model weights
            if freeze:
                for i in range(6):
                    for param in self.conv_model.features[i].parameters():
                        param.requires_grad = False

        else:
            self.conv_model = convnext_tiny()
            
        self.conv_model.features = nn.Sequential(*self.conv_model.features[:6])

        self.pos_encoder = PositionalEncoding(512, dropout=0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,  # This should match the feature size of ConvNeXt's last layer
            nhead=8,      # Number of attention heads
            dim_feedforward=1024,
            dropout=0.1,
            activation='gelu',
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.conv_model.classifier = torch.nn.Sequential(
            nn.Linear(
                in_features=512,
                out_features=num_classes,
                bias=True
            )
    )

    def forward(self, x):
        x = self.conv_model.features(x)

        b, c, h, w = x.size()
        x = x.view(b, c, h * w).permute(2, 0, 1)

        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # x = x.permute(0, 2, 1).view(b, c, h, w)

        # x = self.conv_model.avgpool(x)

        x = torch.mean(x, dim=0)

        x = self.conv_model.classifier(x)

        return x