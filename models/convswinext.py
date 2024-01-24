from torchvision.models import convnext_tiny
from torchvision.models import ConvNeXt_Tiny_Weights

from torchvision.models import convnext_base
from torchvision.models import ConvNeXt_Base_Weights

from torchvision.models import convnext_large
from torchvision.models import ConvNeXt_Large_Weights


from typing import Any, Callable, List, Optional
from torchvision.ops.misc import MLP, Permute

from torchvision.utils import _log_api_usage_once
from torchvision.ops.stochastic_depth import StochasticDepth

from torch import nn, Tensor
import torch
import torch.nn.functional as F

from models.test import SwinTransformer

import math



class ConvSwinNeXtBase(nn.Module):
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
            self.conv_model = convnext_base()
            
        self.conv_model.features = nn.Sequential(*self.conv_model.features[:6])
        self.transformer = SwinTransformer(img_size = 16, window_size = 8, patch_size = 1, in_chans = 512)

        # self.pos_encoder = PositionalEncoding(512, dropout=0.2)

        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=512,  # This should match the feature size of ConvNeXt's last layer
        #     nhead=8,      # Number of attention heads
        #     dim_feedforward=2048,
        #     dropout=0.2,
        #     activation='gelu',
        #     batch_first=False
        # )
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)



        self.conv_model.classifier = torch.nn.Sequential(
            nn.Linear(
                in_features=768,
                out_features=num_classes,
                bias=True
            )
    )

    def forward(self, x):
        


        x = self.conv_model.features(x)

        x = torch.nn.functional.pad(x, (1,1,1,1), "constant",0)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        # b, c, h, w = x.size()
        # x = x.view(b, c, h * w).permute(2, 0, 1)

        

        # x = self.pos_encoder(x)




        # x = self.transformer_encoder(x)




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






