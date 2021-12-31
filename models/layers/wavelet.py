import torch
import torch.nn as nn
import math
import pickle

class WaveletTransform(nn.Module): 
    """
    https://github.com/hhb072/WaveletSRNet/blob/master/networks.py#L27
    
    for usage see https://github.com/hhb072/WaveletSRNet/blob/f0219900056c505143d9831b44a112453784b2a7/main.py#L111
    
    ```python
    wavelet_dec = WaveletTransform(scale=opt.upscale, dec=True)
    wavelet_rec = WaveletTransform(scale=opt.upscale, dec=False)  
    criterion_m = nn.MSELoss(size_average=True)
    ...
    # test
    wavelets = srnet(input)
    prediction = wavelet_rec(wavelets)
    mse = criterion_m(prediction, target)
    psnr = 10 * log10(1 / (mse.data[0]) )
    
    ...
    # train
    
    target_wavelets = wavelet_dec(target)
    
    batch_size = target.size(0)
    wavelets_lr = target_wavelets[:,0:3,:,:]
    wavelets_sr = target_wavelets[:,3:,:,:]
    
    wavelets_predict = srnet(input)
    img_predict = wavelet_rec(wavelets_predict)
    
    
    loss_lr = loss_MSE(wavelets_predict[:,0:3,:,:], wavelets_lr, opt.mse_avg)
    loss_sr = loss_MSE(wavelets_predict[:,3:,:,:], wavelets_sr, opt.mse_avg)
    loss_textures = loss_Textures(wavelets_predict[:,3:,:,:], wavelets_sr)
    loss_img = loss_MSE(img_predict, target, opt.mse_avg)
    
    loss = loss_sr.mul(0.99) + loss_lr.mul(0.01) + loss_img.mul(0.1) + loss_textures.mul(1)    
    ```
    """
    def __init__(self, scale=1, dec=True, params_path='weights/wavelet_weights_c2.pkl', transpose=True):
        super().__init__()
        
        self.scale = scale
        self.dec = dec
        self.transpose = transpose
        
        ks = int(math.pow(2, self.scale)  )
        nc = 3 * ks * ks
        
        if dec:
          self.conv = nn.Conv2d(in_channels=3, out_channels=nc, kernel_size=ks, stride=ks, padding=0, groups=3, bias=False)
        else:
          self.conv = nn.ConvTranspose2d(in_channels=nc, out_channels=3, kernel_size=ks, stride=ks, padding=0, groups=3, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                f = file(params_path,'rb')
                dct = pickle.load(f)
                f.close()
                m.weight.data = torch.from_numpy(dct['rec%d' % ks])
                m.weight.requires_grad = False  
                           
    def forward(self, x): 
        if self.dec:
          output = self.conv(x)          
          if self.transpose:
            osz = output.size()
            #print(osz)
            output = output.view(osz[0], 3, -1, osz[2], osz[3]).transpose(1,2).contiguous().view(osz)            
        else:
          if self.transpose:
            xx = x
            xsz = xx.size()
            xx = xx.view(xsz[0], -1, 3, xsz[2], xsz[3]).transpose(1,2).contiguous().view(xsz)             
          output = self.conv(xx)        
        return output 