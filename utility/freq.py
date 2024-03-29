import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
from numpy.fft import *
import scipy.signal as signal
from scipy.linalg import toeplitz
from scipy.signal import kaiser


# CZT TRANSFORM --------------------------------------------------------------

def czt(x, M=None, W=None, A=1.0, simple=False, t_method='ce', f_method='std'):
    """Calculate Chirp Z-transform (CZT).
    Using an efficient algorithm. Solves in O(n log n) time.
    See algorithm 1 in Sukhoy & Stoytchev 2019 (full reference in README).
    Args:
        x (np.ndarray): input array
        M (int): length of output array
        W (complex): complex ratio between points
        A (complex): complex starting point
        simple (bool): use simple algorithm?
        t_method (str): Toeplitz matrix multiplication method. 'ce' for 
            circulant embedding, 'pd' for Pustylnikov's decomposition, 'mm'
            for simple matrix multiplication
        f_method (str): FFT method. 'std' for standard FFT (from NumPy), or 
            'fast' for a method that may be faster for large arrays. Warning:
            'fast' doesn't seem to work very well. More testing required.
    Returns:
        np.ndarray: Chirp Z-transform
    """
    
    N = len(x)
    if M is None:
        M = N
    if W is None:
        W = np.exp(-2j * np.pi / M)
        
    if simple:
        k = np.arange(M)
        X = np.zeros(M, dtype=complex)
        z = A * W ** -k
        for n in range(N):
            X += x[n] * z ** -n
        return X

    if f_method == 'fast':
        print("Warning: 'fast' doesn't seem to work very well. More testing needed.")

    k = np.arange(N)
    X = W ** (k ** 2 / 2) * A ** -k * x
    r = W ** (-(k ** 2) / 2)
    k = np.arange(M)
    c = W ** (-(k ** 2) / 2)
    if t_method.lower() == 'ce':
        X = _toeplitz_mult_ce(r, c, X, f_method)
    elif t_method.lower() == 'pd':
        X = _toeplitz_mult_pd(r, c, X, f_method)
    elif t_method.lower() == 'mm':
        X = np.matmul(toeplitz(r, c), X)
    else:
        print("t_method not recognized.")
        raise ValueError
    for k in range(M):
        X[k] = W ** (k ** 2 / 2) * X[k]

    return X


def iczt(X, N=None, W=None, A=1.0, t_method='ce', f_method='std'):
    """Calculate inverse Chirp Z-transform (ICZT).
    Args:
        X (np.ndarray): input array
        N (int): length of output array
        W (complex): complex ratio between points
        A (complex): complex starting point
        t_method (str): Toeplitz matrix multiplication method. 'ce' for 
            circulant embedding, 'pd' for Pustylnikov's decomposition, 'mm'
            for simple matrix multiplication
        f_method (str): FFT method. 'std' for standard FFT (from NumPy), or 
            'fast' for a method that may be faster for large arrays. Warning:
            'fast' doesn't seem to work very well. More testing required.
    Returns:
        np.ndarray: Inverse Chirp Z-transform
    """

    M = len(X)
    if N is None:
        N = M
    if W is None:
        W = np.exp(-2j * np.pi / M)

    return np.conj(czt(np.conj(X), M=N, W=W, A=A, t_method=t_method, f_method=f_method)) / M


# OTHER TRANSFORMS -----------------------------------------------------------

def dft(t, x, f=None):
    """Convert signal from time-domain to frequency-domain using a Discrete 
    Fourier Transform (DFT).
    Args:
        t (np.ndarray): time
        x (np.ndarray): time-domain signal
        f (np.ndarray): frequency for output signal
    Returns:
        np.ndarray: frequency-domain signal
    """

    if f is None:
        dt = t[1] - t[0]  # time step
        Fs = 1 / dt       # sample frequency
        f = np.linspace(-Fs / 2, Fs / 2, len(t))

    X = np.zeros(len(f), dtype=complex)
    for k in range(len(X)):
        X[k] = np.sum(x * np.exp(-2j * np.pi * f[k] * t))

    return f, X


def idft(f, X, t=None):
    """Convert signal from time-domain to frequency-domain using an Inverse 
    Discrete Fourier Transform (IDFT).
    Args:
        f (np.ndarray): frequency
        X (np.ndarray): frequency-domain signal
        t (np.ndarray): time for output signal
    Returns:
        np.ndarray: time-domain signal
    """

    if t is None:
        bw = f.max() - f.min()
        t = np.linspace(0, bw / 2, len(f))

    N = len(t)
    x = np.zeros(N, dtype=complex)
    for n in range(len(x)):
        x[n] = np.sum(X * np.exp(2j * np.pi * f * t[n]))
        # for k in range(len(X)):
        #     x[n] += X[k] * np.exp(2j * np.pi * f[k] * t[n])
    x /= N

    return t, x


# FREQ <--> TIME-DOMAIN CONVERSION -------------------------------------------

def time2freq(t, x, f=None, f_orig=None):
    """Convert signal from time-domain to frequency-domain.
    Args:
        t (np.ndarray): time
        x (np.ndarray): time-domain signal
        f (np.ndarray): frequency for output signal
        f_orig (np.ndarray): frequency sweep of the original signal, necessary
            for normalization if the new frequency sweep is different from the
            original
    Returns:
        np.ndarray: frequency-domain signal
    """

    # Input time array
    t1, t2 = t.min(), t.max()  # start / stop time
    dt = t[1] - t[0]           # time step
    Nt = len(t)                # number of time points
    Fs = 1 / dt                # sampling frequency

    # Output frequency array
    if f is None:
        f = np.linspace(-Fs / 2, Fs / 2, Nt)
    f1, f2 = f.min(), f.max()  # start / stop
    df = f[1] - f[0]           # frequency step
    bw = f2 - f1               # bandwidth
    Nf = len(f)                # number of frequency points

    # Correction factor (normalization)
    if f_orig is not None:
        k = 1 / (dt * (f_orig.max() - f_orig.min()))
    else:
        k = 1 / (dt * (f.max() - f.min()))

    # Step
    W = np.exp(-2j * np.pi * bw / (Nf - 1) / Fs)

    # Starting point
    A = np.exp(2j * np.pi * f1 / Fs)

    # Frequency-domain transform
    freq_data = czt(x, Nf, W, A)

    return f, freq_data / k


def freq2time(f, X, t=None, t_orig=None):
    """Convert signal from frequency-domain to time-domain.
    Args:
        f (np.ndarray): frequency
        X (np.ndarray): frequency-domain signal
        t (np.ndarray): time for output signal
    Returns:
        np.ndarray: time-domain signal
    """

    # Input frequency
    f1, f2 = f.min(), f.max()  # start / stop frequency
    df = f[1] - f[0]           # frequency step
    bw = f2 - f1               # bandwidth
    fc = (f1 + f2) / 2         # center frequency
    Nf = len(f)                # number of frequency points
    t_alias = 1 / df           # alias-free interval

    # Output time
    if t is None:
        t = np.linspace(0, t_alias, Nf)
    t1, t2 = t.min(), t.max()  # start / stop time
    dt = t[1] - t[0]           # time step
    Nt = len(t)                # number of time points
    Fs = 1 / dt                # sampling frequency

    # Correction factor (normalization)
    if t_orig is not None:
        k = (t.max() - t.min()) / df / (t_orig.max() - t_orig.min()) **2
    else:
        k = 1

    # Step
    W = np.exp(-2j * np.pi * bw / (Nf - 1) / Fs)

    # Starting point
    A = np.exp(2j * np.pi * t1 / t_alias)

    # Time-domain transform
    time_data = iczt(X, N=Nt, W=W, A=A)

    # Phase shift
    n = np.arange(len(time_data))
    phase = np.exp(2j * np.pi * f1 * n * dt)
    # phase = np.exp(2j * np.pi * (f1 + df / 2) * n * dt)

    return t, time_data * phase / k


# WINDOW ---------------------------------------------------------------------

def get_window(f, f_start=None, f_stop=None, beta=6):
    """Get Kaiser-Bessel window.
    See: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.kaiser.html
    
    Args:
        f (np.ndarray): frequency 
        f_start (float): start frequency
        f_stop (float): stop frequency
        beta (float): KB parameter for Kaiser Bessel filter
    Returns:
        np.ndarray: windowed S-parameter
    """

    # Frequency limits
    if f_start is None:
        f_start = f.min()
    if f_stop is None:
        f_stop = f.max()

    # Get corresponding indices
    idx_start = np.abs(f - f_start).argmin()
    idx_stop  = np.abs(f - f_stop).argmin()
    idx_span  = idx_stop - idx_start

    # Make window
    window = np.r_[np.zeros(idx_start),
                    kaiser(idx_span, beta),
                    np.zeros(len(f) - idx_stop)]

    return window 


def window(f, s, f_start=None, f_stop=None, beta=6, normalize=True):
    """Window frequency-domain data using Kaiser-Bessel filter.
    
    Args:
        f (np.ndarray): frequency 
        s (np.ndarray): S-parameter
        f_start (float): start frequency
        f_stop (float): stop frequency
        beta (float): KB parameter for Kaiser Bessel filter
    Returns:
        np.ndarray: windowed S-parameter
    """

    _window = get_window(f, f_start=f_start, f_stop=f_stop, beta=beta)

    # Normalize
    if normalize:
        w0 = np.mean(_window)
    else:
        w0 = 1

    return s * _window / w0


# HELPER FUNCTIONS -----------------------------------------------------------

def _toeplitz_mult_ce(r, c, x, f_method='std'):
    """Multiply Toeplitz matrix by vector using circulant embedding.
    
    "Compute the product y = Tx of a Toeplitz matrix T and a vector x, where T
    is specified by its first row r = (r[0], r[1], r[2],...,r[N-1]) and its 
    first column c = (c[0], c[1], c[2],...,c[M-1]), where r[0] = c[0]."
    
    See algorithm S1 in Sukhoy & Stoytchev 2019 (full reference in README).
    
    Args:
        r (np.ndarray): first row of Toeplitz matrix
        c (np.ndarray): first column of Toeplitz matrix
        x (np.ndarray): vector to multiply the Toeplitz matrix
        f_method (str): FFT method. 'std' for standard FFT (from NumPy), or 
            'fast' for a method that may be faster for large arrays.
    
    Returns:
        np.ndarray: product of Toeplitz matrix and vector x
        
    """
    N = len(r)
    M = len(c)
    assert r[0] == c[0]
    assert len(x) == N
    n = int(2 ** np.ceil(np.log2(M + N - 1)))
    assert n >= M
    assert n >= N
    chat = np.r_[c, np.zeros(n - (M + N - 1)), r[-(N-1):][::-1]]
    xhat = _zero_pad(x, n)
    yhat = _circulant_multiply(chat, xhat, f_method)
    y = yhat[:M]
    return y


def _toeplitz_mult_pd(r, c, x, f_method='std'):
    """Multiply Toeplitz matrix by vector using Pustylnikov's decomposition.
    
    Compute the product y = Tx of a Toeplitz matrix T and a vector x, where T
    is specified by its first row r = (r[0], r[1], r[2],...,r[N-1]) and its 
    first column c = (c[0], c[1], c[2],...,c[M-1]), where r[0] = c[0].
    
    See algorithm S3 in Sukhoy & Stoytchev 2019 (full reference in README).
    
    Args:
        r (np.ndarray): first row of Toeplitz matrix
        c (np.ndarray): first column of Toeplitz matrix
        x (np.ndarray): vector to multiply the Toeplitz matrix
        f_method (str): FFT method. 'std' for standard FFT (from NumPy), or 
            'fast' for a method that may be faster for large arrays.
    
    Returns:
        np.ndarray: product of Toeplitz matrix and vector x
        
    """
    N = len(r)
    M = len(c)
    assert r[0] == c[0]
    assert len(x) == N
    n = int(2 ** np.ceil(np.log2(M + N - 1)))
    if N != n:
        r = _zero_pad(r, n)
        x = _zero_pad(x, n)
    if M != n:
        c = _zero_pad(c, n)
    c1 = np.empty(n, dtype=complex)
    c2 = np.empty(n, dtype=complex)
    c1[0] = 0.5 * c[0]
    c2[0] = 0.5 * c[0]
    for k in range(1, n):
        c1[k] = 0.5 * (c[k] + r[n - k])
        c2[k] = 0.5 * (c[k] - r[n - k])
    y1 = _circulant_multiply(c1, x, f_method)
    y2 = _skew_circulant_multiply(c2, x, f_method)
    y = y1[:M] + y2[:M]
    return y


def _zero_pad(x, n):
    """Zero pad an array x to length n by appending zeros.
    
    See algorithm S2 in Sukhoy & Stoytchev 2019 (full reference in README).
    
    Args:
        x (np.ndarray): array x
        n (int): length of output array
        
    Returns:
        np.ndarray: array x with padding
        
    """
    m = len(x)
    assert m <= n
    xhat = np.zeros(n, dtype=complex)
    xhat[:m] = x
    return xhat


def _circulant_multiply(c, x, f_method='std'):
    """Multiply a circulat matrix by a vector.
    
    Compute the product y = Gx of a circulat matrix G and a vector x, where G 
    is generated by its first column c=(c[0], c[1],...,c[n-1]).
    Runs in O(n log n) time.
    See algorithm S4 in Sukhoy & Stoytchev 2019 (full reference in README).
    
    Args:
        c (np.ndarray): first column of circulant matrix G
        x (np.ndarray): vector x
        f_method (str): FFT method. 'std' for standard FFT (from NumPy), or 
            'fast' for a method that may be faster for large arrays.
    Returns:
        np.ndarray: product Gx
    
    """
    n = len(c)
    assert len(x) == n
    if f_method == 'std':
        C = np.fft.fft(c)
        X = np.fft.fft(x)
        Y = np.empty(n, dtype=complex)
        for k in range(n):
            Y[k] = C[k] * X[k]
        y = np.fft.ifft(Y)
    elif f_method.lower() == 'fast':
        C = _fft(c)
        X = _fft(x)
        Y = np.empty(n, dtype=complex)
        for k in range(n):
            Y[k] = C[k] * X[k]
        y = _ifft(Y)
    else:
        print("f_method not recognized.")
        raise ValueError
    return y


def _skew_circulant_multiply(c, x, f_method='std'):
    """Multiply a skew-circulant matrix by a vector.
    
    Runs in O(n log n) time.
    
    See algorithm S7 in Sukhoy & Stoytchev 2019 (full reference in README).
    
    Args:
        c (np.ndarray): first column of skew-circulant matrix G
        x (np.ndarray): vector x
        f_method (str): FFT method. 'std' for standard FFT (from NumPy), or 
            'fast' for a method that may be faster for large arrays.
    Returns:
        np.ndarray: product Gx
    """
    n = len(c)
    assert len(x) == n
    chat = np.empty(n, dtype=complex)
    xhat = np.empty(n, dtype=complex)
    for k in range(n):
        chat[k] = c[k] * np.exp(-1j * k * np.pi / n)
        xhat[k] = x[k] * np.exp(-1j * k * np.pi / n)
    # k = np.arange(n, dtype=complex)
    # chat = c * np.exp(-1j * k * np.pi / n)
    # xhat = c * np.exp(-1j * k * np.pi / n)
    y = _circulant_multiply(chat, xhat, f_method)
    for k in range(n):
        y[k] = y[k] * np.exp(1j * k * np.pi / n)
    return y


def _fft(x):
    """FFT algorithm. Runs in O(n log n) time.
    Args:
        x (np.ndarray): input
    Returns:
        np.ndarray: FFT of x
    """
    n = len(x)
    if n == 1:
        return x
    xe = x[0::2]
    xo = x[1::2]
    y1 = np.fft.fft(xe)
    y2 = np.fft.fft(xo)
    # Todo: simplify
    y = np.zeros(n, dtype=complex)
    for k in range(n // 2 - 1):
        w = np.exp(-2j * np.pi * k / n)
        y[k]            = y1[k] + w * y2[k]
        y[k + (n // 2)] = y1[k] - w * y2[k]
    return y


def _ifft(y):
    """IFFT algorithm. Runs in O(n log n) time.
    Args:
        y (np.ndarray): input
    Returns:
        np.ndarray: IFFT of y
    """
    n = len(y)
    if n == 1:
        return y
    ye = y[0::2]
    yo = y[1::2]
    x1 = np.fft.ifft(ye)
    x2 = np.fft.ifft(yo)
    # TODO: simplify
    x = np.zeros(n, dtype=complex)
    for k in range(n // 2 -1):
        w = np.exp(2j * np.pi * k / n)
        x[k]            = (x1[k] + w * x2[k]) / 2
        x[k + (n // 2)] = (x1[k] - w * x2[k]) / 2
    return x




########################################################
### FFT BASED FILTERS ###########################
#########################################

def bandpass_filter(im: np.ndarray, 
                    band_center=0.3, 
                    band_width=0.1, 
                    sample_spacing=None, 
                    mask=None):
    '''Bandpass filter the image (assumes the image is square)
    
    Returns
    -------
    im_bandpass: np.ndarray
    mask: np.ndarray
        if mask is present, use this mask to set things to 1 instead of bandpass
    '''
    
    # find freqs
    if sample_spacing is None: # use normalized freqs [-1, 1]
        freq_arr = fftshift(fftfreq(n=im.shape[0]))
        freq_arr /= np.max(np.abs(freq_arr))
    else:
        sample_spacing = 0.8 # arcmins
        freq_arr = fftshift(fftfreq(n=im.shape[0], d=sample_spacing)) # 1 / arcmin
        # print(freq_arr[0], freq_arr[-1])

    # go to freq domain
    im_f = fftshift(fft2(im))
    '''
    plt.imshow(np.abs(im_f))
    plt.xlabel('frequency x')
    plt.ylabel('frequency y')
    '''

    # bandpass
    if mask is not None:
        assert mask.shape == im_f.shape, 'mask shape does not match shape in freq domain'
        mask_bandpass = mask
    else:
        mask_bandpass = np.zeros(im_f.shape)
        for r in range(im_f.shape[0]):
            for c in range(im_f.shape[1]):
                dist = np.sqrt(freq_arr[r]**2 + freq_arr[c]**2)
                if dist > band_center - band_width / 2 and dist < band_center + band_width / 2:
                    mask_bandpass[r, c] = 1


    im_f_masked = np.multiply(im_f, mask_bandpass)
    '''
    plt.imshow(np.abs(im_f_masked))
    plt.xticks([0, 127.5, 255], labels=[freq_arr[0].round(2), 0, freq_arr[-1].round(2)])
    plt.yticks([0, 127.5, 255], labels=[freq_arr[0].round(2), 0, freq_arr[-1].round(2)])
    plt.show()
    '''

    im_bandpass = np.real(ifft2(ifftshift(im_f_masked)))
    return im_bandpass

'''
Written by Alan Dong
Based on Jae S. Lim, "Two Dimensional Signal and Image Processing" 1990
'''
def bandpass_filter_norm_fast(im: np.ndarray, cutoff_low=0.25, cutoff_high=0.75, kernel_length = 25):
    '''Return bandpass-filtered image, with freqs normalized
    '''
    
    def ftrans2(b: np.ndarray, t=None):
        '''Implements McClellan transform which produces 2D filter from 1D filter
        Params
        ------
        b - 1D filter        
        t - transform matrix, defaults to McClellan transformation
        '''
        if len(b.squeeze().shape) > 1:
            raise Exception("ftrans2: b must be a one dimensional array!")
        elif np.all(b == 0):
            raise Exception("ftrans2: b must have at least one nonzero element!")
        elif len(b) % 2 == 0:
            raise Exception("ftrans2: b must be odd length!")
        elif np.any( abs(b-b[::-1]) > np.sqrt(np.finfo(b.dtype).eps) ):
            raise Exception("ftrans2: b must be symmetric!")

        if t is None:
            t = np.array([[1.,2,1],[2,-4,2],[1,2,1]]) / 8. # McClellan transformation
        n = (len(b)-1)//2
        b = np.fft.ifftshift(b)
        a = np.concatenate([[b[0]], 2.0*b[1:n+1]])

        inset = np.floor((np.array(t.shape)-1)/2).astype("int")

        # Use Chebyshev polynomials to compute h
        P0 = 1
        P1 = t
        h = a[1] * P1
        rows = np.array([inset[0]])
        cols = np.array([inset[1]])
        h[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1] = h[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1] + a[0] * P0
        for i in range(2,n+1):
            P2 = 2 * signal.convolve2d(t, P1)
            rows = rows + inset[0]
            cols = cols + inset[1]
            P2[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1] = P2[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1] - P0
            rows = inset[0] + np.arange(P1.shape[0])
            cols = inset[1] + np.arange(P1.shape[1])
            hh = h
            h = a[i] * P2
            h[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1] = h[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1] + hh
            P0 = P1
            P1 = P2
        return h

    def filter2(im: np.ndarray, h: np.ndarray):
        '''2D filtering
        Params
        ------
        im - image to be filtered
        h - 2D filter
        '''
        if np.issubdtype(im.dtype, np.integer):
            im = im.astype("float")
        if len(im.shape) == 2:
            out = signal.convolve2d(im, h, "same")
        elif len(im.shape) == 3:
            out = np.zeros(im.shape, dtype=im.dtype)
            for i in range(im.shape[2]):
                out[...,i] = signal.convolve2d(im[...,i], h, "same")
        else:
            raise Exception("filter2: im must be two or three dimensional!")
        return out    
    
    
    
    b = signal.firwin(kernel_length, cutoff=[cutoff_low, cutoff_high], window='blackmanharris', pass_zero=False)
    h = ftrans2(b)
    return filter2(im, h)
