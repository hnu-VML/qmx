"""
Definition of the BiFastDVDnet model

Copyright (C) 2019, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

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
        # # training
        # out_forward = torch.sign(x)
        # mask1 = x < -1
        # mask2 = x < 0
        # mask3 = x < 1
        # out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        # out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        # out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        # out = out_forward.detach() - out3.detach() + out3
        
        # testing
        out = torch.sign(x)

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
        # scaling_factor = scaling_factor.detach()
        # # stx()
        # binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        # binary_weights_no_grad = torch.sign(real_weights)
        # stx()
        # cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        # binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        
        # ------------- test -------------
        binary_weights = scaling_factor * torch.sign(real_weights)
        
        y = F.conv2d(x, binary_weights, self.bias, stride=self.stride, padding=self.padding, groups=self.groups)

        return y

class BinaryConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, groups=1):
        super(BinaryConv2d, self).__init__()
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

class BinaryConv2d_Down(nn.Module):
    '''
    input: b,c,h,w
    output: b,c/2,2h,2w
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False, groups=1):
        super(BinaryConv2d_Down, self).__init__()
        self.biconv_1 = BinaryConv2d(in_channels, in_channels, kernel_size, stride, padding, bias, groups)
        self.biconv_2 = BinaryConv2d(in_channels, in_channels, kernel_size, stride, padding, bias, groups)
        self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)

    def forward(self, x):
        '''
        x: b,c,h,w
        out: b,2c,h/2,w/2
        '''
        out = self.avg_pool(x)
        out_1 = out
        out_2 = out_1.clone()
        out_1 = self.biconv_1(out_1)
        out_2 = self.biconv_2(out_2)
        out = torch.cat([out_1, out_2], dim=1)

        return out

class BinaryConv2d_Up(nn.Module):
    '''
    input: b,c,h,w
    output: b,c/2,2h,2w
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False, groups=1):
        super(BinaryConv2d_Up, self).__init__()
        self.biconv_1 = BinaryConv2d(out_channels, out_channels, kernel_size, stride, padding, bias, groups)
        self.biconv_2 = BinaryConv2d(out_channels, out_channels, kernel_size, stride, padding, bias, groups)

    def forward(self, x):
        '''
        x: b,c,h,w
        out: b,c/2,2h,2w
        '''
        b,c,h,w = x.shape
        out = F.interpolate(x, scale_factor=2, mode='bilinear')
        
        out_1 = out[:,:c//2,:,:]
        out_2 = out[:,c//2:,:,:]

        out_1 = self.biconv_1(out_1)
        out_2 = self.biconv_2(out_2)

        out = (out_1 + out_2) / 2

        return out

class InputCvBlock(nn.Module):
	'''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''
	def __init__(self, num_in_frames, out_ch, ncolor):
		super(InputCvBlock, self).__init__()
		self.interm_ch = 30
		self.convblock = nn.Sequential(
			nn.Conv2d(num_in_frames*(ncolor+1), num_in_frames*self.interm_ch, \
					  kernel_size=3, padding=1, groups=num_in_frames, bias=False),
			nn.BatchNorm2d(num_in_frames*self.interm_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(num_in_frames*self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.convblock(x)

class OutputCvBlock(nn.Module):
	'''Conv2d => BN => ReLU => Conv2d'''
	def __init__(self, in_ch, out_ch):
		super(OutputCvBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(in_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
		)

	def forward(self, x):
		return self.convblock(x)

class BiCvBlock(nn.Module):
	'''(Conv2d => BN => ReLU) x 2'''
	def __init__(self, in_ch, out_ch):
		super(BiCvBlock, self).__init__()
		self.convblock = nn.Sequential(
			BinaryConv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
			BinaryConv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1)
		)

	def forward(self, x):
		return self.convblock(x)

class BiDownBlock(nn.Module):
	'''Downscale + (Conv2d => BN => ReLU)*2'''
	def __init__(self, in_ch, out_ch):
		super(BiDownBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
   			# RPReLU(out_ch),
			BiCvBlock(out_ch, out_ch)
		)

	def forward(self, x):
		return self.convblock(x)

class BiUpBlock(nn.Module):
	'''(Conv2d => BN => ReLU)*2 + Upscale'''
	def __init__(self, in_ch, out_ch):
		super(BiUpBlock, self).__init__()
		self.convblock = nn.Sequential(
			BiCvBlock(in_ch, in_ch),
			nn.Conv2d(in_ch, out_ch*4, kernel_size=3, padding=1, bias=False),
			nn.PixelShuffle(2)
		)

	def forward(self, x):
		return self.convblock(x)

class BiDenBlock(nn.Module):
	""" Definition of the denosing block of BiFastDVDnet.
	Inputs of constructor:
		num_input_frames: int. number of input frames
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=3, num_color_channels=3):
		super(BiDenBlock, self).__init__()
		self.chs_lyr0 = 32
		self.chs_lyr1 = 64
		self.chs_lyr2 = 128

		self.inc = InputCvBlock(num_in_frames=num_input_frames, out_ch=self.chs_lyr0, ncolor=num_color_channels)
		self.downc0 = BiDownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		self.downc1 = BiDownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
		self.upc2 = BiUpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
		self.upc1 = BiUpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
		self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=num_color_channels)

		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, in0, in1, in2, noise_map):
		'''Args:
			inX: Tensor, [N, C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Input convolution block
		x0 = self.inc(torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim=1))
		# Downsampling
		x1 = self.downc0(x0)
		x2 = self.downc1(x1)
		# Upsampling
		x2 = self.upc2(x2)
		x1 = self.upc1(x1+x2)
		# Estimation
		x = self.outc(x0+x1)

		# Residual
		x = in1 - x

		return x

class BiFastDVDnet(nn.Module):
	""" Definition of the BiFastDVDnet model.
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=5, num_color_channels=3):
		super(BiFastDVDnet, self).__init__()
		self.num_input_frames = num_input_frames
		# Define models of each denoising stage
		# self.spatial = spatialDnCNN(num_input_frames=1, num_color_channels=num_color_channels)
		self.temp1 = BiDenBlock(num_input_frames=3, num_color_channels=num_color_channels)
		self.temp2 = BiDenBlock(num_input_frames=3, num_color_channels=num_color_channels)
		# Init weights
		self.reset_params()
		self.num_color_channels = num_color_channels

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, x, noise_map):
		'''Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		C = self.num_color_channels # number of color channels
		# Unpack inputs
		(x0, x1, x2, x3, x4) = tuple(x[:, m*C:m*C+C, :, :] for m in range(self.num_input_frames))

		# First stage
		x20 = self.temp1(x0, x1, x2, noise_map)
		x21 = self.temp1(x1, x2, x3, noise_map)
		x22 = self.temp1(x2, x3, x4, noise_map)

		#Second stage
		x = self.temp2(x20, x21, x22, noise_map)

		return x
	
	def __nchannel__(self):
		return self.num_color_channels
