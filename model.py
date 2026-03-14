import torch
import torch.nn as nn
import torchvision.models as models

class FrameOrderModel(nn.Module):

    def __init__(self):
        super().__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=2048,
                nhead=8,
                batch_first=True
            ),
            num_layers=2
        )

        self.head = nn.Linear(2048,1)

    def forward(self,x):

        B,T,C,H,W = x.shape

        x = x.view(B*T,C,H,W)

        feat = self.encoder(x)
        feat = feat.view(B,T,2048)

        feat = self.transformer(feat)

        scores = self.head(feat).squeeze(-1)

        return scores