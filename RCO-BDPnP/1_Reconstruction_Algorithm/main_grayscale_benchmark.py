import os
import cv2
import time
import math
import h5py
import torch
import argparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from statistics import mean

from packages.pnp_sci import admmdenoise_cacti,admmdenoise_cacti_batch
from utils import (A_, At_, A_Batch, At_Batch)
from packages.bifastdvdnet.models import BiFastDVDnet
from packages.fastdvdnet.models import FastDVDnet
from packages.ffdnet.models import FFDNet
from scipy.io.matlab.mio import _open_file
from scipy.io.matlab.miobase import get_matfile_version

def Reconstruction_single(args, dataname, method, sigma=[100/255, 50/255, 25/255, 12/255], 
                          iter_max=[20, 20, 20, 20], MMCO=False, nframe=None):
    
    matfile = os.path.join(args.datasetdir, dataname)
    # load data
    if get_matfile_version(_open_file(matfile, appendmat=True)[0])[0] < 2: # MATLAB .mat v7.2 or lower versions
        file = sio.loadmat(matfile) # for '-v7.2' and lower version of .mat file (MATLAB)
        meas = np.float32(file['meas'])
        mask = np.float32(file['mask'])
        orig = np.float32(file['orig'])
    else: # MATLAB .mat v7.3
        file =  h5py.File(matfile, 'r')  # for '-v7.3' .mat file (MATLAB)
        meas = np.float32(file['meas']).transpose()
        mask = np.float32(file['mask']).transpose()
        orig = np.float32(file['orig']).transpose()

    print("-----------------------", dataname, "-----------------------")
    # print("meas shape:", meas.shape, " mask shape:", mask.shape)

    if nframe == None:
        nframe = meas.shape[2]
    
    if args.is_real:
        MAXB = 255/nframe*2
    else:
        MAXB = 255

    # common parameters and pre-calculation for PnP
    # define forward model and its transpose
    if MMCO:
        A  = lambda x : A_Batch(x, mask) # batch forward model function handle
        At = lambda y : At_Batch(y, mask) # batch transpose of forward model
    else:
        A  = lambda x : A_(x, mask) # forward model function handle
        At = lambda y : At_(y, mask) # transpose of forward model
    

    mask_sum = np.sum(mask, axis=2)
    mask_sum[mask_sum==0] = 1

    # pre-load the model for fastdvdnet image denoising
    NUM_IN_FR_EXT = 5 # temporal size of patch
    
    if method=="gap-tv":
        args.sigma = None
        model = None
    elif method=="pnp-ffdnet":
        in_ch = 1
        model_fn = 'packages/ffdnet/models/net_gray.pth'
        # Create model
        model = FFDNet(num_input_channels=in_ch).cuda()
        # Load saved weights
        state_dict = torch.load(model_fn)
        # model.load_state_dict(state_dict)
        model.load_state_dict({k.replace('module.',''): v for k,v in state_dict.items()})
        model.eval()
    elif method=="pnp-fastdvdnet":
        model = FastDVDnet(num_input_frames=NUM_IN_FR_EXT,num_color_channels=1).cuda()
        # Load saved weights
        state_temp_dict = torch.load('./packages/fastdvdnet/model_gray.pth')
        model.load_state_dict(state_temp_dict)
        model.eval()
    elif ((method=="pnp-bifastdvdnet") or (method=="rco-bdpnp")):
        model = BiFastDVDnet(num_input_frames=NUM_IN_FR_EXT,num_color_channels=1).cuda()
        # Load saved weights
        state_temp_dict = torch.load('./packages/bifastdvdnet/logs/simulation.pth') #photomechanical_imaging
        model.load_state_dict({k.replace('module.',''): v for k,v in state_temp_dict['state_dict'].items()})
        model.eval()
    else:
        raise ValueError('Unsupported method {}!'.format(method))

    if MMCO:
        vgapfastdvdnet,tgapfastdvdnet,psnr_gapfastdvdnet,ssim_gapfastdvdnet = admmdenoise_cacti_batch(meas, mask, A, At,
                                            v0=None, orig=orig, nframe=nframe,
                                            MAXB=MAXB, method=method, model=model, 
                                            iter_max=iter_max, sigma=sigma,
                                            tv_weight=args.tv_weight, tv_iter_max=args.tv_iter_max,
                                            MMCO = MMCO)
    else:
        vgapfastdvdnet,tgapfastdvdnet,psnr_gapfastdvdnet,ssim_gapfastdvdnet = admmdenoise_cacti(meas, mask, A, At,
                                            v0=None, orig=orig, nframe=nframe,
                                            MAXB=MAXB, method=method, model=model, 
                                            iter_max=iter_max, sigma=sigma,
                                            tv_weight=args.tv_weight, tv_iter_max=args.tv_iter_max,
                                            MMCO = MMCO)

    print('{} PSNR {:2.2f} dB, SSIM {:.4f}, running time {:.1f} seconds.'.format(
            method.upper(), mean(psnr_gapfastdvdnet), mean(ssim_gapfastdvdnet), tgapfastdvdnet))

    
    if not os.path.exists(args.resultsdir + '/target'):
        os.mkdir(args.resultsdir + '/target')
    if not os.path.exists(args.resultsdir + '/reconstruction'):
        os.mkdir(args.resultsdir + '/reconstruction')
    if not os.path.exists(args.resultsdir + '/target/' + dataname.split('.')[0] + "_" + method):
        os.makedirs(args.resultsdir + '/target/' + dataname.split('.')[0] + "_" + method)
    if not os.path.exists(args.resultsdir + '/reconstruction/' + dataname.split('.')[0] + "_" + method):
        os.makedirs(args.resultsdir + '/reconstruction/' + dataname.split('.')[0] + "_" + method)

    for i in range(nframe*mask.shape[2]):
        cv2.imwrite(args.resultsdir + '/' + 'target/' + dataname.split('.')[0] + "_" + method + '/'+ 'out_' + str(i).zfill(8) + '.png', orig[:,:,i])
        cv2.imwrite(args.resultsdir + '/' + 'reconstruction/' + dataname.split('.')[0] + "_" + method + '/'+ 'out_' + str(i).zfill(8) + '.png', vgapfastdvdnet[:,:,i]*255)
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Video SCI Reconstruction")

    #Training parameters
    parser.add_argument("--method", type=str, default="rco-bdpnp", 	\
                        help="SCI reconstruction model, such as gap-tv, pnp-ffdnet, pnp-fastdvdnet, pnp-bifastdvdnet and rco-bdpnp.")
    parser.add_argument("--gpu", type=str, default="0", \
                        help="Select the GPU for acceleration")
    parser.add_argument("--is_real", type=bool, default=False, \
                        help="Whether to process real data")
    parser.add_argument("--tv_weight", type=float, default=0.3, \
                        help="TV denoising weight (larger for smoother but slower)")
    parser.add_argument("--tv_iter_max", type=int, default=5, \
                        help="TV denoising maximum number of iterations each")
    parser.add_argument("--datasetdir", type=str, default="./dataset/cacti/grayscale_benchmark", \
                        help="Path to the Data to be Reconstructed")
    parser.add_argument("--resultsdir", type=str, default="./results_grayscale", \
                        help="Path to Save the Reconstructed Results")
    argspar = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = argspar.gpu
    
    if not os.path.exists(argspar.resultsdir):
        os.mkdir(argspar.resultsdir)
    
    # # --------------------------------------- GAP-TV --------------------------------------- #
    # print("--------------------------------------- GAP-TV ---------------------------------------")
    # Reconstruction_single(argspar, "aerial32_cacti.mat"  , method="gap-tv")
    # Reconstruction_single(argspar, "crash32_cacti.mat"   , method="gap-tv")
    # Reconstruction_single(argspar, "drop40_cacti.mat"    , method="gap-tv", nframe=1)
    # Reconstruction_single(argspar, "kobe32_cacti.mat"    , method="gap-tv")
    # Reconstruction_single(argspar, "runner40_cacti.mat"  , method="gap-tv", nframe=1)
    # Reconstruction_single(argspar, "traffic48_cacti.mat" , method="gap-tv")
    
    # ------------------------------------ PnP-FFDVDnet ------------------------------------ #
    print("------------------------------------ PnP-FFDVDnet ------------------------------------")
    Reconstruction_single(argspar, "aerial32_cacti.mat",  method="pnp-ffdnet",
                          sigma=[50/255,25/255,12/255,6/255], iter_max=[10,10,10,10])
    Reconstruction_single(argspar, "crash32_cacti.mat",   method="pnp-ffdnet",
                          sigma=[50/255,25/255,12/255,6/255], iter_max=[10,10,10,10])
    Reconstruction_single(argspar, "drop40_cacti.mat",    method="pnp-ffdnet",
                          sigma=[50/255,25/255,12/255,6/255], iter_max=[10,10,10,10], nframe=1)
    Reconstruction_single(argspar, "kobe32_cacti.mat",    method="pnp-ffdnet",
                          sigma=[50/255,25/255,12/255,6/255], iter_max=[10,10,10,10])
    Reconstruction_single(argspar, "runner40_cacti.mat",  method="pnp-ffdnet",
                          sigma=[50/255,25/255,12/255,6/255], iter_max=[10,10,10,10], nframe=1)
    Reconstruction_single(argspar, "traffic48_cacti.mat", method="pnp-ffdnet",
                          sigma=[50/255,25/255,12/255,6/255], iter_max=[10,10,10,10])
    
    # ----------------------------------- PnP-FastDVDnet ----------------------------------- #
    print("----------------------------------- PnP-FastDVDnet -----------------------------------")
    Reconstruction_single(argspar, "aerial32_cacti.mat",  method="pnp-fastdvdnet")
    Reconstruction_single(argspar, "crash32_cacti.mat",   method="pnp-fastdvdnet")
    Reconstruction_single(argspar, "drop40_cacti.mat",    method="pnp-fastdvdnet", nframe=1)
    Reconstruction_single(argspar, "kobe32_cacti.mat",    method="pnp-fastdvdnet")
    Reconstruction_single(argspar, "runner40_cacti.mat",  method="pnp-fastdvdnet", nframe=1)
    Reconstruction_single(argspar, "traffic48_cacti.mat", method="pnp-fastdvdnet")
    
    # ---------------------------------- PnP-BiFastDVDnet ---------------------------------- #
    print("------------------------------------ PnP-BiFastDVDnet ------------------------------------")
    Reconstruction_single(argspar, "aerial32_cacti.mat",  method="pnp-bifastdvdnet", 
                          sigma=[145/255,45/255,20/255,20/255],  iter_max=[20,20,20,20], MMCO=False, nframe=None)
    Reconstruction_single(argspar, "crash32_cacti.mat",   method="pnp-bifastdvdnet", 
                          sigma=[145/255,100/255,50/255,25/255], iter_max=[20,20,20,20], MMCO=False, nframe=None)
    Reconstruction_single(argspar, "drop40_cacti.mat",    method="pnp-bifastdvdnet", 
                          sigma=[150/255,60/255,20/255,9/255],   iter_max=[20,20,20,20], MMCO=False, nframe=1)
    Reconstruction_single(argspar, "kobe32_cacti.mat",    method="pnp-bifastdvdnet", 
                          sigma=[100/255,50/255,25/255,15/255],  iter_max=[20,20,20,20], MMCO=False, nframe=None)
    Reconstruction_single(argspar, "runner40_cacti.mat",  method="pnp-bifastdvdnet", 
                          sigma=[100/255,65/255,30/255,6/255],   iter_max=[50,10,10,10], MMCO=False, nframe=1)
    Reconstruction_single(argspar, "traffic48_cacti.mat", method="pnp-bifastdvdnet", 
                          sigma=[145/255,100/255,50/255,25/255], iter_max=[20,20,20,20], MMCO=False, nframe=None)
    
    # ------------------------------------- RCO-BDPnP ------------------------------------- #
    print("------------------------------------ RCO-BDPnP ------------------------------------")
    Reconstruction_single(argspar, "aerial32_cacti.mat",  method="rco-bdpnp", 
                          sigma=[145/255,45/255,20/255,20/255],  iter_max=[20,20,20,20], MMCO=True,  nframe=None)
    Reconstruction_single(argspar, "crash32_cacti.mat",   method="rco-bdpnp", 
                          sigma=[145/255,100/255,50/255,25/255], iter_max=[20,20,20,20], MMCO=True,  nframe=None)
    Reconstruction_single(argspar, "drop40_cacti.mat",    method="rco-bdpnp", 
                          sigma=[150/255,60/255,20/255,9/255],   iter_max=[20,20,20,20], MMCO=True,  nframe=1)
    Reconstruction_single(argspar, "kobe32_cacti.mat",    method="rco-bdpnp", 
                          sigma=[100/255,50/255,25/255,15/255],  iter_max=[20,20,20,20], MMCO=True,  nframe=None)
    Reconstruction_single(argspar, "runner40_cacti.mat",  method="rco-bdpnp", 
                          sigma=[100/255,65/255,30/255,6/255],   iter_max=[50,10,10,10], MMCO=False, nframe=1)
    Reconstruction_single(argspar, "traffic48_cacti.mat", method="rco-bdpnp", 
                          sigma=[145/255,100/255,50/255,25/255], iter_max=[20,20,20,20], MMCO=True,  nframe=None)