import time
import math
import skimage
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
# from packages.vnlnet.test import vnlnet
from packages.ffdnet.test_ffdnet_ipol import (ffdnet_vdenoiser, ffdnet_rgb_denoise)
from packages.biffdnet.test_biffdnet_ipol import (biffdnet_vdenoiser,)
from packages.bifastdvdnet.test_bifastdvdnet import bifastdvdnet_denoiser
from packages.fastdvdnet.test_fastdvdnet import fastdvdnet_denoiser
# from packages.colour_demosaicing.bayer import demosaicing_CFA_Bayer_bilinear as demosaicing_bayer
from colour_demosaicing.bayer import demosaicing_CFA_Bayer_Menon2007 as demosaicing_bayer
from utils import (A_, At_, psnr)
if skimage.__version__ < '0.18':
    from skimage.measure import (compare_psnr, compare_ssim)
else: # skimage.measure deprecated in version 0.18 ( -> skimage.metrics )
    import skimage.metrics.peak_signal_noise_ratio as compare_psnr
    import skimage.metrics.structural_similarity   as compare_ssim



def admmdenoise_cacti(meas, mask, A, At, v0=None, orig=None, iframe=0, nframe=1, MAXB=1., 
                      maskdirection='plain', HAF=False, CMTC=False, is_real=False, **args):
    '''
    Alternating direction method of multipliers (ADMM) or generalized 
    alternating projection (GAP) -based denoising (based on the 
    plug-and-play (PnP) framework) algorithms for video snapshot compressive
    imaging (SCI) or coded aperture compressive temporal imaging (CACTI, 
    Llull et al. Opt. Express 2013).
    '''
    nmask = mask.shape[-1]

    mask_sum = np.sum(mask, axis=tuple(range(2,mask.ndim)))
    mask_sum[mask_sum==0] = 1

    x_ = np.zeros((*mask.shape[:-1],nmask*nframe), dtype=np.float32)
    psnr_, ssim_, psnrall_ = ([], [], [])
    begin_time = time.time()
    # loop over all the coded frames [nframe]
    for kf in range(nframe):
        print('%s Reconstruction coded frame block %2d of %2d ...'
              %(args['method'].upper(), kf+1, nframe))
        if orig is not None:
            orig_k = orig[...,(kf+iframe)*nmask:(kf+iframe+1)*nmask]/MAXB
        meas_k = meas[...,kf+iframe]/MAXB
        if v0 is None:
            v0_k = None
        else: # initialization according to the direction of the masks [up as calibration]
            v0_k = v0[:,:,kf*nmask:(kf+1)*nmask]
            if (maskdirection.lower() == 'updown' and (kf+iframe) % 2 == 1) or \
               (maskdirection.lower() == 'downup' and (kf+iframe) % 2 == 0):  # down (up as mask)
               v0_k = v0_k[...,::-1]

        x_k, psnr_k, ssim_k =  gap_denoise(meas_k, mask_sum, A, At, x0=v0_k, X_orig=orig_k, 
                                           HAF=HAF, CMTC = CMTC, is_real=is_real, **args)
        
        if (maskdirection.lower() == 'updown' and (kf+iframe) % 2 == 1) or \
           (maskdirection.lower() == 'downup' and (kf+iframe) % 2 == 0):   # down (up as mask)
            x_k = x_k[...,::-1]
            psnr_k = psnr_k[::-1]
            ssim_k = ssim_k[::-1]
        
        t_ = time.time() - begin_time
        x_[...,kf*nmask:(kf+1)*nmask] = x_k
        psnr_.extend(psnr_k)
        ssim_.extend(ssim_k)
        
    return x_, t_, psnr_, ssim_

def admmdenoise_cacti_batch(meas, mask, A, At, v0=None, orig=None, nframe=1, 
                            MAXB=1., HAF=False, CMTC=False, is_real=False, **args):
    '''
    Alternating direction method of multipliers (ADMM) or generalized 
    alternating projection (GAP) -based denoising (based on the 
    plug-and-play (PnP) framework) algorithms for video snapshot compressive
    imaging (SCI) or coded aperture compressive temporal imaging (CACTI, 
    Llull et al. Opt. Express 2013).
    '''
    nmask = mask.shape[-1]

    mask_sum = np.sum(mask, axis=tuple(range(2,mask.ndim)))
    mask_sum[mask_sum==0] = 1
    
    begin_time = time.time()
    # loop over all the coded frames [nframe]

    if v0 is None:
        v0_k = None

    x, psnr, ssim =  gap_denoise_batch(meas[:,:,0:nframe]/MAXB, mask_sum, A, At, nmask, 
                                       x0=v0_k, X_orig=orig[:,:,0:nframe*nmask]/MAXB, 
                                       HAF=HAF, CMTC = CMTC, is_real=is_real, **args)
        
    t = time.time() - begin_time
        
    return x, t, psnr, ssim


def gap_denoise(y, Phi_sum, A, At, _lambda=1, accelerate=True, 
                method='tv', iter_max=50, noise_estimate=False, sigma=None, 
                tv_weight=0.1, tv_iter_max=5, multichannel=True, x0=None, 
                X_orig=None, model=None, show_iqa=True, HAF=False, CMTC=False, is_real=False):
    '''
    Alternating direction method of multipliers (ADMM)[1]-based denoising 
    regularization for snapshot compressive imaging (SCI).

    Parameters
    ----------
    y : two-dimensional (2D) ndarray of ints, uints or floats
        Input single measurement of the snapshot compressive imager (SCI).
    Phi : three-dimensional (3D) ndarray of ints, uints or floats, omitted
        Input sensing matrix of SCI with the third dimension as the 
        time-variant, spectral-variant, volume-variant, or angular-variant 
        masks, where each mask has the same pixel resolution as the snapshot
        measurement.
    Phi_sum : 2D ndarray,
        Sum of the sensing matrix `Phi` along the third dimension.
    A : function
        Forward model of SCI, where multiple encoded frames are collapsed into
        a single measurement.
    At : function
        Transpose of the forward model.
    proj_meth : {'admm' or 'gap'}, optional
        Projection method of the data term. Alternating direction method of 
        multipliers (ADMM)[1] and generalizedv alternating projection (GAP)[2]
        are used, where ADMM for noisy data, especially real data and GAP for 
        noise-free data.
    gamma : float, optional
        Parameter in the ADMM projection, where more noisy measurements require
        greater gamma.
    method : string, optional
        method used as the regularization imposing on the prior term of the 
        reconstruction.
    _lambda : float, optional
        Regularization factor balancing the data term and the prior term, 
        where larger `_lambda` imposing more constrains on the prior term. 
    iter_max : int or uint, optional 
        Maximum number of iterations.
    accelerate : boolean, optional
        Enable acceleration in GAP.
    noise_estimate : boolean, optional
        Enable noise estimation in the denoiser.
    sigma : one-dimensional (1D) ndarray of ints, uints or floats
        Input noise standard deviation for the denoiser if and only if noise 
        estimation is disabled(i.e., noise_estimate==False). The scale of sigma 
        is [0, 255] regardless of the the scale of the input measurement and 
        masks.
    tv_weight : float, optional
        weight in total variation (TV) denoising.
    x0 : 3D ndarray 
        Start point (initialized value) for the iteration process of the 
        reconstruction.
    model : pretrained model for image/video denoising.

    Returns
    -------
    x : 3D ndarray
        Reconstructed 3D scene captured by the SCI system.

    References
    ----------
    .. [1] X. Liao, H. Li, and L. Carin, "Generalized Alternating Projection 
           for Weighted-$\ell_{2,1}$ Minimization with Applications to 
           Model-Based Compressive Sensing," SIAM Journal on Imaging Sciences, 
           vol. 7, no. 2, pp. 797-823, 2014.
    .. [2] X. Yuan, "Generalized alternating projection based total variation 
           minimization for compressive sensing," in IEEE International 
           Conference on Image Processing (ICIP), 2016, pp. 2539-2543.
    .. [3] Y. Liu, X. Yuan, J. Suo, D. Brady, and Q. Dai, "Rank Minimization 
           for Snapshot Compressive Imaging," IEEE Transactions on Pattern 
           Analysis and Machine Intelligence, doi:10.1109/TPAMI.2018.2873587, 
           2018.

    Code credit
    -----------
    Xin Yuan, Bell Labs, xyuan@bell-labs.com, created Aug 7, 2018.
    Yang Liu, Tsinghua University, y-liu16@mails.tsinghua.edu.cn, 
      updated Jan 22, 2019.

    See Also
    --------
    admm_denoise
    '''
    # [0] initialization
    if x0 is None:
        # x0 = At(y, Phi) # default start point (initialized value)
        x0 = At(y) # default start point (initialized value)
    if not isinstance(sigma, list):
        sigma = [sigma]
    if not isinstance(iter_max, list):
        iter_max = [iter_max] * len(sigma)
    # y1 = np.zeros(y.shape)
    y1 = np.zeros_like(y) 
    # [1] start iteration for reconstruction
    x = x0 # initialization
    psnr_all = []
    k = 0
    for idx, nsig in enumerate(sigma): # iterate all noise levels
        for it in range(iter_max[idx]):
            yb = A(x)
            if accelerate: # accelerated version of GAP
                y1 = y1 + (y-yb)
                x = x + _lambda*(At((y1-yb)/Phi_sum)) # GAP_acc
            else:
                x = x + _lambda*(At((y-yb)/Phi_sum)) # GAP
            
            # # Observing the actual noise distribution
            # if k%5 == 0:
            #     # np.save('x.npy', x)
            #     # np.save('X_orig.npy', X_orig)
            #     real_noise = x - X_orig

            #     # Create a histogram of the error distribution
            #     plt.hist(real_noise.flatten(), bins=200, alpha=0.75, color='blue')
            #     plt.xlabel('Error')
            #     plt.ylabel('Frequency')
            #     plt.title('Error Distribution')

            #     # Save the histogram as an image
            #     plt.savefig("error_distribution_histogram"+str(k)+"_before.png")
            
            # switch denoiser 
            if method.lower() == 'gap-tv': # total variation (TV) denoising
                x = denoise_tv_chambolle(x, tv_weight, n_iter_max=tv_iter_max, 
                                         multichannel=multichannel)
            elif method.lower() == 'pnp-ffdnet': # FFDNet frame-wise video denoising
                if HAF and is_real:
                    if  idx == 0:
                        x = ffdnet_vdenoiser(x, nsig, model)
                    elif idx == 1:
                        x = denoise_tv_chambolle(x, tv_weight, n_iter_max=tv_iter_max, 
                                                    multichannel=multichannel)
                    else:
                        x = ffdnet_vdenoiser(x, nsig, model)
                elif HAF and not is_real:
                    if  idx == 0:
                        x = denoise_tv_chambolle(x, tv_weight, n_iter_max=tv_iter_max, 
                                                    multichannel=multichannel)
                    else:
                        x = ffdnet_vdenoiser(x, nsig, model)
                else:
                    x = ffdnet_vdenoiser(x, nsig, model)
            elif method.lower() == 'pnp-biffdnet': # FFDNet frame-wise video denoising
                if HAF and is_real:
                    if  idx == 0:
                        x = biffdnet_vdenoiser(x, nsig, model)
                    elif idx == 1:
                        x = denoise_tv_chambolle(x, tv_weight, n_iter_max=tv_iter_max, 
                                                    multichannel=multichannel)
                    else:
                        x = biffdnet_vdenoiser(x, nsig, model)
                elif HAF and not is_real:
                    if  idx == 0:
                        x = denoise_tv_chambolle(x, tv_weight, n_iter_max=tv_iter_max, 
                                                    multichannel=multichannel)
                    else:
                        x = biffdnet_vdenoiser(x, nsig, model)
                else:
                    x = biffdnet_vdenoiser(x, nsig, model)
            elif ((method.lower()=="pnp-fastdvdnet") or (method.lower()=="pnp-fastdvdnet-cmtc") or (method.lower()=="pnp-fastdvdnet-cmtc-haf") or (method.lower()=="pnp-fastdvdnet-haf")): # FastDVDnet video denoising
                if HAF and is_real:
                    if  idx == 0:
                        x = fastdvdnet_denoiser(x, nsig, model, gray=True, \
                                                  CMTC=CMTC) # grayscale video denoising
                    elif idx == 1:
                        x = denoise_tv_chambolle(x, tv_weight, n_iter_max=tv_iter_max, 
                                                 multichannel=multichannel)
                    else:
                        x = fastdvdnet_denoiser(x, nsig, model, gray=True, \
                                                  CMTC=CMTC) # grayscale video denoising
                elif HAF and not is_real:
                    if  idx == 0:
                        x = denoise_tv_chambolle(x, tv_weight, n_iter_max=tv_iter_max, 
                                                 multichannel=multichannel)
                    else:
                        x = fastdvdnet_denoiser(x, nsig, model, gray=True, \
                                                  CMTC=CMTC) # grayscale video denoising
                else:
                    x = fastdvdnet_denoiser(x, nsig, model, gray=True, \
                                            CMTC=CMTC) # grayscale video denoising
            elif ((method.lower() == 'pnp-bifastdvdnet') or (method.lower() == 'bdpnp') or (method.lower() == 'pnp-bifastdvdnet-cmtc') or (method.lower() == 'pnp-bifastdvdnet-haf')): # BiFastDVDnet video denoising
                if HAF and is_real:
                    if  idx == 0:
                        x = bifastdvdnet_denoiser(x, nsig, model, gray=True, \
                                                  CMTC=CMTC) # grayscale video denoising
                    elif idx == 1:
                        x = denoise_tv_chambolle(x, tv_weight, n_iter_max=tv_iter_max, 
                                                 multichannel=multichannel)
                    else:
                        x = bifastdvdnet_denoiser(x, nsig, model, gray=True, \
                                                  CMTC=CMTC) # grayscale video denoising
                elif HAF and not is_real:
                    if  idx == 0:
                        x = denoise_tv_chambolle(x, tv_weight, n_iter_max=tv_iter_max, 
                                                 multichannel=multichannel)
                    else:
                        x = bifastdvdnet_denoiser(x, nsig, model, gray=True, \
                                              CMTC=CMTC) # grayscale video denoising
                else:
                    x = bifastdvdnet_denoiser(x, nsig, model, gray=True, \
                                              CMTC=CMTC) # grayscale video denoising
            else:
                raise ValueError('Unsupported denoiser {}!'.format(method))
            
            # # Observing the actual noise distribution
            # if k%5 == 0:
            #     # np.save('x.npy', x)
            #     # np.save('X_orig.npy', X_orig)
            #     real_noise = x - X_orig

            #     # Create a histogram of the error distribution
            #     plt.hist(real_noise.flatten(), bins=200, alpha=0.75, color='blue')
            #     plt.xlabel('Error')
            #     plt.ylabel('Frequency')
            #     plt.title('Error Distribution')

            #     # Save the histogram as an image
            #     plt.savefig("error_distribution_histogram"+str(k)+"_after.png")
                
            # cv2.imwrite("results/" + str(k) + ".jpg", x[:,:,0]*255)
            # [optional] calculate image quality assessment, i.e., PSNR for 
            # every five iterations
            if show_iqa and X_orig is not None:
                psnr_all.append(psnr(X_orig, x))
                if (k+1)%5 == 0:
                    if not noise_estimate and nsig is not None:
                        if nsig < 1:
                            print('  {0} iteration {1: 3d}, sigma {2: 3g}/255, ' 
                            'PSNR {3:2.2f} dB.'.format(method.upper(), 
                            k+1, nsig*255, psnr_all[k]))
                        else:
                            print('  {0} iteration {1: 3d}, sigma {2: 3g}, ' 
                                'PSNR {3:2.2f} dB.'.format(method.upper(), 
                                k+1, nsig, psnr_all[k]))
                    else:
                        print('  {0} iteration {1: 3d}, ' 
                            'PSNR {2:2.2f} dB.'.format(method.upper(), 
                            k+1, psnr_all[k]))
            k = k+1
    
    psnr_ = []
    ssim_ = []
    nmask = x.shape[-1]
    if X_orig is not None:
        for imask in range(nmask):
            psnr_.append(compare_psnr(X_orig[...,imask], x[...,imask], data_range=1.))
            ssim_.append(compare_ssim(X_orig[...,imask], x[...,imask], data_range=1.,multichannel=x[...,imask].ndim>2))
    return x, psnr_, ssim_


def gap_denoise_batch(y, Phi_sum, A, At, nmask, _lambda=1, accelerate=True, 
                      method='tv', iter_max=50, noise_estimate=False, sigma=None, 
                      tv_weight=0.1, tv_iter_max=5, multichannel=True, x0=None, 
                      X_orig=None, model=None, show_iqa=True, HAF=False, CMTC=False, is_real=False):
    '''
    Alternating direction method of multipliers (ADMM)[1]-based denoising 
    regularization for snapshot compressive imaging (SCI).

    Parameters
    ----------
    y : two-dimensional (2D) ndarray of ints, uints or floats
        Input single measurement of the snapshot compressive imager (SCI).
    Phi : three-dimensional (3D) ndarray of ints, uints or floats, omitted
        Input sensing matrix of SCI with the third dimension as the 
        time-variant, spectral-variant, volume-variant, or angular-variant 
        masks, where each mask has the same pixel resolution as the snapshot
        measurement.
    Phi_sum : 2D ndarray,
        Sum of the sensing matrix `Phi` along the third dimension.
    A : function
        Forward model of SCI, where multiple encoded frames are collapsed into
        a single measurement.
    At : function
        Transpose of the forward model.
    proj_meth : {'admm' or 'gap'}, optional
        Projection method of the data term. Alternating direction method of 
        multipliers (ADMM)[1] and generalizedv alternating projection (GAP)[2]
        are used, where ADMM for noisy data, especially real data and GAP for 
        noise-free data.
    gamma : float, optional
        Parameter in the ADMM projection, where more noisy measurements require
        greater gamma.
    denoiser : string, optional
        Denoiser used as the regularization imposing on the prior term of the 
        reconstruction.
    _lambda : float, optional
        Regularization factor balancing the data term and the prior term, 
        where larger `_lambda` imposing more constrains on the prior term. 
    iter_max : int or uint, optional 
        Maximum number of iterations.
    accelerate : boolean, optional
        Enable acceleration in GAP.
    noise_estimate : boolean, optional
        Enable noise estimation in the denoiser.
    sigma : one-dimensional (1D) ndarray of ints, uints or floats
        Input noise standard deviation for the denoiser if and only if noise 
        estimation is disabled(i.e., noise_estimate==False). The scale of sigma 
        is [0, 255] regardless of the the scale of the input measurement and 
        masks.
    tv_weight : float, optional
        weight in total variation (TV) denoising.
    x0 : 3D ndarray 
        Start point (initialized value) for the iteration process of the 
        reconstruction.
    model : pretrained model for image/video denoising.

    Returns
    -------
    x : 3D ndarray
        Reconstructed 3D scene captured by the SCI system.

    References
    ----------
    .. [1] X. Liao, H. Li, and L. Carin, "Generalized Alternating Projection 
           for Weighted-$\ell_{2,1}$ Minimization with Applications to 
           Model-Based Compressive Sensing," SIAM Journal on Imaging Sciences, 
           vol. 7, no. 2, pp. 797-823, 2014.
    .. [2] X. Yuan, "Generalized alternating projection based total variation 
           minimization for compressive sensing," in IEEE International 
           Conference on Image Processing (ICIP), 2016, pp. 2539-2543.
    .. [3] Y. Liu, X. Yuan, J. Suo, D. Brady, and Q. Dai, "Rank Minimization 
           for Snapshot Compressive Imaging," IEEE Transactions on Pattern 
           Analysis and Machine Intelligence, doi:10.1109/TPAMI.2018.2873587, 
           2018.

    Code credit
    -----------
    Xin Yuan, Bell Labs, xyuan@bell-labs.com, created Aug 7, 2018.
    Yang Liu, Tsinghua University, y-liu16@mails.tsinghua.edu.cn, 
      updated Jan 22, 2019.

    See Also
    --------
    admm_denoise
    '''
    # [0] initialization
    if x0 is None:
        # x0 = At(y, Phi) # default start point (initialized value)
        # x0 = np.zeros_like(X_orig)
        # for i in range(groupnums):
        #     x0[:,:,i*nmask:(i+1)*nmask] = At(y[:,:,i]) # default start point (initialized value)
        x0 =  At(y)
    if not isinstance(sigma, list):
        sigma = [sigma]
    if not isinstance(iter_max, list):
        iter_max = [iter_max] * len(sigma)

    y1 = np.zeros_like(y) 
    # [1] start iteration for reconstruction
    x = x0 # initialization
    psnr_all = []
    k = 0
    for idx, nsig in enumerate(sigma): # iterate all noise levels
        for it in range(iter_max[idx]):
            yb =  A(x)
            if accelerate: # accelerated version of GAP
                y1 = y1 + (y-yb)
                x = x + _lambda*(At((y1-yb)/Phi_sum[:, :, np.newaxis])) # GAP_acc
            else:
                x = x + _lambda*(At((y1-yb)/Phi_sum[:, :, np.newaxis])) # GAP_acc
            # method 
            if method.lower() == 'gap-tv': # total variation (TV) denoising
                x = denoise_tv_chambolle(x, tv_weight, n_iter_max=tv_iter_max, 
                                         multichannel=multichannel)
            elif method.lower() == 'pnp-ffdnet': # FFDNet frame-wise video denoising
                x = ffdnet_vdenoiser(x, nsig, model)
            elif method.lower() == 'pnp-biffdnet': # FFDNet frame-wise video denoising
                x = biffdnet_vdenoiser(x, nsig, model)
            elif (method.lower()=="pnp-fastdvdnet"): # FastDVDnet video denoising
                if HAF and is_real:
                    if  idx == 0:
                        x = fastdvdnet_denoiser(x, nsig, model, gray=True, \
                                                  CMTC=CMTC) # grayscale video denoising
                    elif idx == 1:
                        x = denoise_tv_chambolle(x, tv_weight, n_iter_max=tv_iter_max, 
                                                 multichannel=multichannel)
                    else:
                        x = fastdvdnet_denoiser(x, nsig, model, gray=True, \
                                                  CMTC=CMTC) # grayscale video denoising
                elif HAF and not is_real:
                    if  idx == 0:
                        x = denoise_tv_chambolle(x, tv_weight, n_iter_max=tv_iter_max, 
                                                 multichannel=multichannel)
                    else:
                        x = fastdvdnet_denoiser(x, nsig, model, gray=True, \
                                                  CMTC=CMTC) # grayscale video denoising
                else:
                    x = fastdvdnet_denoiser(x, nsig, model, gray=True, \
                                        CMTC=CMTC) # grayscale video denoising
            elif (method.lower() == 'bdpnp'): # FastDVDnet video denoising
                if HAF and is_real:
                    if  idx == 0:
                        x = bifastdvdnet_denoiser(x, nsig, model, gray=True, \
                                                  CMTC=CMTC) # grayscale video denoising
                    elif idx == 1:
                        x = denoise_tv_chambolle(x, tv_weight, n_iter_max=tv_iter_max, 
                                                 multichannel=multichannel)
                    else:
                        x = bifastdvdnet_denoiser(x, nsig, model, gray=True, \
                                                  CMTC=CMTC) # grayscale video denoising
                elif HAF and not is_real:
                    if  idx == 0:
                        x = denoise_tv_chambolle(x, tv_weight, n_iter_max=tv_iter_max, 
                                                 multichannel=multichannel)
                    else:
                        x = bifastdvdnet_denoiser(x, nsig, model, gray=True, \
                                                  CMTC=CMTC) # grayscale video denoising
                else:
                    x = bifastdvdnet_denoiser(x, nsig, model, gray=True, \
                                              CMTC=CMTC) # grayscale video denoising
            else:
                raise ValueError('Unsupported denoiser {}!'.format(method))
            
            # cv2.imwrite("results/" + str(k) + ".jpg", x[:,:,5]*255)
            # [optional] calculate image quality assessment, i.e., PSNR for 
            # every five iterations
            if show_iqa and X_orig is not None:
                psnr_all.append(psnr(X_orig, x))
                if (k+1)%5 == 0:
                    if not noise_estimate and nsig is not None:
                        if nsig < 1:
                            print('  {0} iteration {1: 3d}, sigma {2: 3g}/255, ' 
                            'PSNR {3:2.2f} dB.'.format(method.upper(), 
                            k+1, nsig*255, psnr_all[k]))
                        else:
                            print('  {0} iteration {1: 3d}, sigma {2: 3g}, ' 
                                'PSNR {3:2.2f} dB.'.format(method.upper(), 
                                k+1, nsig, psnr_all[k]))
                    else:
                        print('  {0} iteration {1: 3d}, ' 
                            'PSNR {2:2.2f} dB.'.format(method.upper(), 
                            k+1, psnr_all[k]))
            k = k+1
    
    psnr_ = []
    ssim_ = []
    nmask = x.shape[-1]
    if X_orig is not None:
        for imask in range(nmask):
            psnr_.append(compare_psnr(X_orig[...,imask], x[...,imask], data_range=1.))
            ssim_.append(compare_ssim(X_orig[...,imask], x[...,imask], data_range=1.,multichannel=x[...,imask].ndim>2))
    return x, psnr_, ssim_