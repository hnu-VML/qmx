"""
Definition of the BiFFDNet model and its custom layers

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import torch.nn as nn
from torch.autograd import Variable
from .functions import upsamplefeatures, concatenate_input_noise_map
import torch
import torch.nn.functional as F

# --------------------------------------------- Binarized Basic Units -----------------------------------------------------------------
class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        # stx()
        out = x + self.bias.expand_as(x)
        return out

class RPReLU(nn.Module):
    def __init__(self, inplanes):
        super(RPReLU, self).__init__()
        self.pr_bias0 = LearnableBias(inplanes)
        self.pr_prelu = nn.PReLU(inplanes)
        self.pr_bias1 = LearnableBias(inplanes)

    def forward(self, x):
        x = self.pr_bias1(self.pr_prelu(self.pr_bias0(x)))      
        return x

class ReDistribution(nn.Module):
    def __init__(self, out_chn):
        super(ReDistribution, self).__init__()
        self.b = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)
        self.k = nn.Parameter(torch.ones(1,out_chn,1,1), requires_grad=True)
    
    def forward(self, x):
        out = x * self.k.expand_as(x) + self.b.expand_as(x)
        return out

class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        # training
        out_forward = torch.sign(x)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3
        
        # # testing
        # out = torch.sign(x)

        return out

class HardBinaryConv(nn.Conv2d):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, groups=1, bias=True):
        super(HardBinaryConv, self).__init__(
            in_chn,
            out_chn,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias
        )

    def forward(self, x):
        real_weights = self.weight
        
        # real_weights_np = real_weights.cpu().detach().numpy()
		# # 绘制张量的分布图
        # plt.figure(figsize=(10, 6))
        # plt.hist(real_weights_np.flatten(), bins=1000, alpha=0.7, color='blue', edgecolor='black')
        # plt.title('Distribution of Tensor Elements')
        # plt.xlabel('Value')
        # plt.ylabel('Frequency')

		# # 保存分布图为图片
        # plt.savefig('origin_distribution.png')
        
        # bw = real_weights - real_weights.view(real_weights.size(0), -1).mean(-1).view(real_weights.size(0), 1, 1, 1)
        # bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        # sw = torch.pow(torch.tensor([2]*bw.size(0)).cuda().float(), (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(bw.size(0), 1, 1, 1).detach()
        # bw = bw * sw
        
        # bw_np = bw.cpu().detach().numpy()
		# # 绘制张量的分布图
        # plt.figure(figsize=(10, 6))
        # plt.hist(bw_np.flatten(), bins=1000, alpha=0.7, color='blue', edgecolor='black')
        # plt.title('Distribution of Tensor Elements')
        # plt.xlabel('Value')
        # plt.ylabel('Frequency')

		# # 保存分布图为图片
        # plt.savefig('norm_distribution.png')
        
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        # ------------- train -------------
        scaling_factor = scaling_factor.detach()
        # stx()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        binary_weights_no_grad = torch.sign(real_weights)
        # stx()
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        
        # # ------------- test -------------
        # binary_weights = scaling_factor * torch.sign(real_weights)
        
        y = F.conv2d(x, binary_weights, self.bias, stride=self.stride, padding=self.padding, groups=self.groups)

        return y

class RBCU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, groups=1):
        super(RBCU, self).__init__()
        self.move0 = ReDistribution(in_channels)
        self.binary_activation = BinaryActivation()
        self.binary_conv = HardBinaryConv(in_chn=in_channels,
        out_chn=out_channels,
        kernel_size=kernel_size,
        stride = stride,
        padding=padding,
        bias=bias,
        groups=groups)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu=RPReLU(in_channels)
        self.ins = None

    def forward(self, x):
        self.ins = x
        out = self.move0(x)
        out = self.binary_activation(out)
        out = self.binary_conv(out)
        out = self.bn(out)
        out = self.relu(out)
        out = out + x
        return out

class UpSampleFeatures(nn.Module):
	r"""Implements the last layer of BiFFDNet
	"""
	def __init__(self):
		super(UpSampleFeatures, self).__init__()
	def forward(self, x):
		# return functions.upsamplefeatures(x)
		return upsamplefeatures(x)

class BiIntermediateDnCNN(nn.Module):
	r"""Implements the middel part of the BiFFDNet architecture, which
	is basically a DnCNN net
	"""
	def __init__(self, input_features, middle_features, num_conv_layers):
		super(BiIntermediateDnCNN, self).__init__()
		self.kernel_size = 3
		self.padding = 1
		self.input_features = input_features
		self.num_conv_layers = num_conv_layers
		self.middle_features = middle_features
		if self.input_features == 5:
			self.output_features = 4 #Grayscale image
		elif self.input_features == 15:
			self.output_features = 12 #RGB image
		else:
			raise Exception('Invalid number of input features')

		layers = []
		layers.append(nn.Conv2d(in_channels=self.input_features,\
								out_channels=self.middle_features,\
								kernel_size=self.kernel_size,\
								padding=self.padding,\
								bias=False))
		# layers.append(nn.ReLU(inplace=True))
		layers.append(RPReLU(self.middle_features))
		for _ in range(0,3):
			layers.append(nn.Conv2d(in_channels=self.middle_features,\
									out_channels=self.middle_features,\
									kernel_size=self.kernel_size,\
									padding=self.padding,\
									bias=False))
			layers.append(nn.BatchNorm2d(self.middle_features))
			layers.append(nn.ReLU(inplace=True))
		for _ in range(3, 9):
			layers.append(RBCU(in_channels=self.middle_features,\
									out_channels=self.middle_features,\
									kernel_size=self.kernel_size,\
									padding=self.padding,\
									bias=False))
		for _ in range(9, self.num_conv_layers-2):
			layers.append(nn.Conv2d(in_channels=self.middle_features,\
									out_channels=self.middle_features,\
									kernel_size=self.kernel_size,\
									padding=self.padding,\
									bias=False))
			layers.append(nn.BatchNorm2d(self.middle_features))
			layers.append(nn.ReLU(inplace=True))
   
		layers.append(nn.Conv2d(in_channels=self.middle_features,\
								out_channels=self.output_features,\
								kernel_size=self.kernel_size,\
								padding=self.padding,\
								bias=False))
		self.itermediate_dncnn = nn.Sequential(*layers)
	def forward(self, x):
		out = self.itermediate_dncnn(x)
		return out

class BiFFDNet(nn.Module):
	r"""Implements the BiFFDNet architecture
	"""
	def __init__(self, num_input_channels):
		super(BiFFDNet, self).__init__()
		self.num_input_channels = num_input_channels
		if self.num_input_channels == 1:
			# Grayscale image
			self.num_feature_maps = 64
			self.num_conv_layers = 15
			self.downsampled_channels = 5
			self.output_features = 4
		elif self.num_input_channels == 3:
			# RGB image
			self.num_feature_maps = 96
			self.num_conv_layers = 12
			self.downsampled_channels = 15
			self.output_features = 12
		else:
			raise Exception('Invalid number of input features')

		self.intermediate_dncnn = BiIntermediateDnCNN(\
				input_features=self.downsampled_channels,\
				middle_features=self.num_feature_maps,\
				num_conv_layers=self.num_conv_layers)
		self.upsamplefeatures = UpSampleFeatures()

	def forward(self, x, noise_sigma):
		# concat_noise_x = functions.concatenate_input_noise_map(\
		# 		x.data, noise_sigma.data)
		concat_noise_x = concatenate_input_noise_map(\
				x.data, noise_sigma.data)
		concat_noise_x = Variable(concat_noise_x)
		h_dncnn = self.intermediate_dncnn(concat_noise_x)
		pred_noise = self.upsamplefeatures(h_dncnn)
		return pred_noise
