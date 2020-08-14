import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import numpy as np


#---------------------------------------------------------------------------------------------
#                The proposed gated skip connections (Fig1 in the paper)
#---------------------------------------------------------------------------------------------
class Residual_Soft_Connection(nn.Module):
    def __init__(self, input_dim, batch_size):
        super(Residual_Soft_Connection, self).__init__()

        self.Alfa  = nn.Parameter(torch.randn(batch_size, input_dim, 1, 1))
        self.Alfa.requires_grad = True

        self.BN1   = nn.BatchNorm2d(input_dim)
        self.Conv1 = nn.Conv2d(in_channels=input_dim, out_channels=int(input_dim/2), kernel_size=(3,3), stride=1, padding=1)
        self.BN2   = nn.BatchNorm2d(int(input_dim/2))
        self.Conv2 = nn.Conv2d(in_channels=int(input_dim/2), out_channels=int(input_dim/4), kernel_size=(3,3), stride=1, padding=1)
        self.BN3   = nn.BatchNorm2d(int(input_dim/4))
        self.Conv3 = nn.Conv2d(in_channels=int(input_dim/4), out_channels=int(input_dim/4), kernel_size=(3,3), stride=1, padding=1)

    def forward(self, X):
        # a*Xl
        scaled_factor = X * self.Alfa

        # F(Xl, Wl)
        out1 = self.BN1(X)
        out1 = F.relu(out1)
        out1 = self.Conv1(out1)

        out2 = self.BN2(out1)
        out2 = F.relu(out2)
        out2 = self.Conv2(out2)

        out3 = self.BN3(out2)
        out3 = F.relu(out3)
        out3 = self.Conv3(out3)

        # Concate 3 outs
        concat_out = torch.cat((out1, out2, out2), dim=1)

        # generate output : out = a*Xl + F(Xl, Wl)
        return concat_out + scaled_factor


#---------------------------------------------------------------------------------------------
#                                The Encoder Module
#---------------------------------------------------------------------------------------------
class Encoder_Module(nn.Module):
    def __init__(self, input_dim, batch_size, down_scale=1):
        super(Encoder_Module, self).__init__()

        self.residual = Residual_Soft_Connection(input_dim, batch_size)
        self.MaxPool  = nn.MaxPool2d(down_scale)


    def forward(self, X):
        Residual_out = self.residual(X)
        Pooling_out  = self.MaxPool(Residual_out)

        return Residual_out, Pooling_out


#---------------------------------------------------------------------------------------------
#       The Decoder Module with Proposed method for Decoder Input (Fig2.b in the paper)
#---------------------------------------------------------------------------------------------
class Decoder_Module(nn.Module):
    def __init__(self, input_dim, batch_size, up_scale=1):
        super(Decoder_Module, self).__init__()

        self.residual = Residual_Soft_Connection(input_dim, batch_size)
        self.UpSample = nn.UpsamplingNearest2d(scale_factor=up_scale)
        self.Conv = nn.Conv2d(in_channels=int(2*input_dim), out_channels=input_dim, kernel_size=(3,3), stride=1, padding=1)

    def forward(self, prev_layer, skip_connection):

        up_out = self.UpSample(prev_layer)
        concat = torch.cat((up_out, skip_connection), dim=1)
        dec_in = self.Conv(concat)
        output = self.residual(dec_in)

        return output


#---------------------------------------------------------------------------------------------
#                   The Complete HourGlass Module whitout Heatmap
#---------------------------------------------------------------------------------------------
class HG_Module(nn.Module):
    def __init__(self, input_dim, batch_size):
        super(HG_Module, self).__init__()

        self.encoder1 = Encoder_Module(input_dim, batch_size, down_scale=8)  # I/8
        self.encoder2 = Encoder_Module(input_dim, batch_size, down_scale=2)  # I/16
        self.encoder3 = Encoder_Module(input_dim, batch_size, down_scale=2)  # I/32
        self.encoder4 = Encoder_Module(input_dim, batch_size, down_scale=2)  # I/64

        self.SkipCon1  = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=(1,1), stride=1, padding=0)
        self.SkipCon2  = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=(1,1), stride=1, padding=0)
        self.SkipCon3  = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=(1,1), stride=1, padding=0)
        self.SkipCon4  = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=(1,1), stride=1, padding=0)

        self.decoder1 = Decoder_Module(input_dim, batch_size, up_scale=2)    # I/32
        self.decoder2 = Decoder_Module(input_dim, batch_size, up_scale=2)    # I/16
        self.decoder3 = Decoder_Module(input_dim, batch_size, up_scale=2)    # I/8
        self.decoder4 = Decoder_Module(input_dim, batch_size, up_scale=8)    # I

    def forward(self, X):
        # Encoder Parts
        Residual_out1, Pooling_out1 = self.encoder1(X)
        Residual_out2, Pooling_out2 = self.encoder2(Pooling_out1)
        Residual_out3, Pooling_out3 = self.encoder2(Pooling_out2)
        Residual_out4, Pooling_out4 = self.encoder2(Pooling_out3)

        # Skip Connection Parts
        Skip_con_out1 = self.SkipCon1(Residual_out1)
        Skip_con_out2 = self.SkipCon2(Residual_out2)
        Skip_con_out3 = self.SkipCon3(Residual_out3)
        Skip_con_out4 = self.SkipCon4(Residual_out4)

        # Decoder Parts
        Decoder_out1 = self.decoder1(Pooling_out4, Skip_con_out4)
        Decoder_out2 = self.decoder2(Decoder_out1, Skip_con_out3)
        Decoder_out3 = self.decoder3(Decoder_out2, Skip_con_out2)
        Decoder_out4 = self.decoder4(Decoder_out3, Skip_con_out1)

        return Decoder_out4


#---------------------------------------------------------------------------------------------
#                   The Complete Model with 4 Stack HG
#---------------------------------------------------------------------------------------------
class Complete_Model(nn.Module):
    def __init__(self, N, batch_size, n_stacks, num_of_joints=14):
        super(Complete_Model, self).__init__()
        self.n_stacks = n_stacks
        self.inConv = nn.Conv2d(in_channels=3, out_channels=N, kernel_size=(3,3), stride=1, padding=1)
        self.HG_Modules = nn.ModuleList([
            HG_Module(N, batch_size) for _ in range(n_stacks)
        ])
        self.Conv_Modules = nn.ModuleList([
            nn.Conv2d(in_channels=N, out_channels=num_of_joints, kernel_size=(1,1), stride=1, padding=0) for _ in range(n_stacks)
        ])


    def forward(self, image):
        # the dimention of image should be [batch_size, channel(3), W, H] -> based on the paper [24,3,256,256]
        #-------------------------------------------------------------------------------------------------
        # First HG in the Stack
        image = self.inConv(image) # input image = [B,3,256,256] -> output = [B,N,256,256]
        heatmaps = []
        for i in range(self.n_stacks):
            image = self.HG_Modules[i](image)
            heatmap = self.Conv_Modules[i](image)
            heatmaps.append(heatmap)
        print(heatmap)

        return torch.cat(heatmaps, dim=1)


class GaussianFilterLayer(nn.Module):
    def __init__(self, n_layers, filter_size, sigma, groups=14):
        super(GaussianFilterLayer, self).__init__()
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(int(filter_size/2)),
            nn.Conv2d(
                n_layers,
                n_layers,
                filter_size,
                stride=1,
                padding=0,
                bias=None,
                groups=groups
            )
        )
        self.weights_init(filter_size, sigma)
        self.requires_grad_(False)

    def forward(self, x):
        return self.seq(x)

    def weights_init(self, filter_size, sigma):
        filter_mat = np.zeros((filter_size , filter_size))
        filter_mat[int(filter_size/2), int(filter_size/2)] = 1
        k = gaussian_filter(filter_mat ,sigma=sigma)
        for name, param in self.named_parameters():
            param.data.copy_(torch.as_tensor(k))
