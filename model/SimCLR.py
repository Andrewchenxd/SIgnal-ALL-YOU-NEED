import torch.nn as nn
import torchvision
import torch
from simclr.modules.resnet_hacks import modify_resnet_model
from simclr.modules.identity import Identity
import torchvision.models as torchvision_models
import warnings
warnings.filterwarnings("ignore")

class SimCLR_Resnet(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, encoder, projection_dim, n_features):
        super(SimCLR_Resnet, self).__init__()

        self.encoder = encoder
        self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.n_features = n_features

        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j

class SimCLR_Vit(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, encoder, projection_dim, n_features,hidden_dim,patch_size):
        super(SimCLR_Vit, self).__init__()

        self.encoder = encoder
        self.encoder.conv_proj = nn.Conv2d(
                in_channels=1, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
            )
        self.n_features = n_features

        # Replace the fc layer with an Identity function
        self.encoder.head = Identity()

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j

class SimCLR_Resnet_fc(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, encoder, numclasses, n_features):
        super(SimCLR_Resnet_fc, self).__init__()

        self.encoder = encoder
        self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.n_features = n_features

        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.Dropout(0.2),
            nn.Linear(self.n_features, numclasses, bias=False),
        )

    def forward(self, x_i):
        h_i = self.encoder(x_i)

        out = self.projector(h_i)
        return out

class SimCLR_Vit_fc(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, encoder, numclasses, n_features,hidden_dim,patch_size):
        super(SimCLR_Vit_fc, self).__init__()

        self.encoder = encoder
        self.encoder.conv_proj = nn.Conv2d(
                in_channels=1, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
            )
        self.n_features = n_features

        # Replace the fc layer with an Identity function
        self.encoder.head = Identity()

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.Dropout(0.2),
            nn.Linear(self.n_features, numclasses, bias=False),
        )

    def forward(self, x_i):
        h_i = self.encoder(x_i)
        out = self.projector(h_i)
        return out

if __name__ == '__main__':
    from simclr.modules import NT_Xent
    from torchvision.models import vit_b_16
    choose='resnet18'
    # encoder=torchvision_models.__dict__['vit_b_16']
    if choose=='resnet18':
        encoder = torchvision_models.__dict__['resnet18'](pretrained=False)
        model=SimCLR_Resnet(encoder,64,512)
        model_fc = SimCLR_Resnet_fc(encoder, 10, 512)
    elif choose=='resnet50':
        encoder = torchvision_models.__dict__['resnet50'](pretrained=True)
        model = SimCLR_Resnet(encoder, 64, 2048)
    elif choose == 'vit_b_32':
        encoder = torchvision_models.__dict__['vit_b_32'](pretrained=True)
        model = SimCLR_Vit(encoder, 64, 1000, 768,32)
    elif choose == 'vit_b_16':
        encoder = torchvision_models.__dict__['vit_b_16'](pretrained=True)

        # encoder = vit_b_16(pretrained=False)
        # # 修改模型的网络层和头的数量
        # encoder.num_layers = 4
        # encoder.num_heads = 6
        model = SimCLR_Vit(encoder, 64, 1000, 768,16)
    elif choose == 'vit_l_32':
        encoder = torchvision_models.__dict__['vit_l_32']
        model = SimCLR_Vit(encoder, 64, 1000, 1024, 32)
    img_size = 224
    channels = 1
    model = model.cuda()
    model_fc=model_fc.cuda()
    images1 = torch.randn((2, channels, img_size, img_size)).cuda()
    images2 = torch.randn((2, channels, img_size, img_size)).cuda()
    h_i, h_j, z_i, z_j = model(images1, images2)
    out=model_fc(images1)
    print(out)
    criterion = NT_Xent(2, 0.07, 1)
    loss = criterion(z_i, z_j)
    print(loss)





