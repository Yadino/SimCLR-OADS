import torch.nn as nn
import torchvision.models as models


class AlexNetSimCLR(nn.Module):

    def __init__(self, out_dim):
        super(AlexNetSimCLR, self).__init__()

        self.backbone = models.alexnet()
        dim_mlp = self.backbone.classifier[6].in_features

        # add mlp projection head
        self.backbone.classifier[6] = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.classifier[6])

    def forward(self, x):
        return self.backbone(x)
