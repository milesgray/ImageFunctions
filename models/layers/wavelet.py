import torch
import torch.nn as nn
import math
import pickle

from .registry import register

@register("dct2d")
class Dct2d(nn.Module):
    """
    Blockwhise 2D DCT
    """
    def __init__(self, blocksize: int=8, interleaving: bool=False):
        """
        Parameters:
        blocksize: int, size of the Blocks for discrete cosine transform 
        interleaving: bool, should the blocks interleave?
        """
        super().__init__() # call super constructor
        
        self.blocksize = blocksize
        self.interleaving = interleaving
        
        if interleaving:
            self.stride = self.blocksize // 2
        else:
            self.stride = self.blocksize
        
        # precompute DCT weight matrix
        A = np.zeros((blocksize,blocksize))
        for i in range(blocksize):
            c_i = 1/np.sqrt(2) if i == 0 else 1.
            for n in range(blocksize):
                A[i,n] = np.sqrt(2/blocksize) * c_i * np.cos((2*n+ 1)/(blocksize*2) * i * np.pi)
        
        # set up conv layer
        self.A = nn.Parameter(torch.tensor(A, dtype=torch.float32), requires_grad=False)
        self.unfold = torch.nn.Unfold(kernel_size=blocksize, padding=0, stride=self.stride)
        
    def forward(self, x: torch.Tensor):
        """
        performs 2D blockwhise DCT
        
        Parameters:
        x: tensor of dimension (N, 1, h, w)
        
        Return:
        tensor of dimension (N, k, blocksize, blocksize)
        where the 2nd dimension indexes the block. Dimensions 3 and 4 are the block DCT coefficients
        """
        
        (N, C, H, W) = x.shape
        assert (C == 1), "DCT is only implemented for a single channel"
        assert (H >= self.blocksize), "Input too small for blocksize"
        assert (W >= self.blocksize), "Input too small for blocksize"
        assert (H % self.stride == 0) and (W % self.stride == 0), "FFT is only for dimensions divisible by the blocksize"
        
        # unfold to blocks
        x = self.unfold(x)
        # now shape (N, blocksize**2, k)
        (N, _, k) = x.shape
        x = x.view(-1, self.blocksize, self.blocksize, k).permute(0,3,1,2)
        # now shape (N, #k, blocksize, blocksize)
        # perform DCT
        coeff = self.A.matmul(x).matmul(self.A.transpose(0,1))
        
        return coeff
    
    def inverse(self, coeff, output_shape):
        """
        performs 2D blockwhise iDCT
        
        Parameters:
        coeff: tensor of dimension (N, k, blocksize, blocksize)
        where the 2nd dimension indexes the block. Dimensions 3 and 4 are the block DCT coefficients
        output_shape: (h, w) dimensions of the reconstructed image
        
        Return:
        tensor of dimension (N, 1, h, w)
        """
        if self.interleaving:
            raise Exception('Inverse block DCT is not implemented for interleaving blocks!')
            
        # perform iDCT
        x = self.A.transpose(0,1).matmul(coeff).matmul(self.A)
        (N, k, _, _) = x.shape
        x = x.permute(0,2,3,1).view(-1, self.blocksize**2, k)
        x = F.fold(x, output_size=(output_shape[-2], output_shape[-1]), kernel_size=self.blocksize, padding=0, stride=self.blocksize)
        return x
    
@register("wavelet_transform")
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
    def __init__(self, scale: int=1, dec: bool=True, params_path: str='weights/wavelet_weights_c2.pkl', transpose=True):
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

    def forward(self, x: torch.Tensor):
        if self.dec:
          output = self.conv(x)
          if self.transpose:
            osz = output.size()
            output = output.view(osz[0], 3, -1, osz[2], osz[3]) \
                            .transpose(1, 2) \
                                .contiguous() \
                                    .view(osz)
        else:
          if self.transpose:
            xx = x
            xsz = xx.size()
            xx = xx.view(xsz[0], -1, 3, xsz[2], xsz[3]) \
                        .transpose(1, 2) \
                            .contiguous() \
                                .view(xsz)
          output = self.conv(xx)
        return output