import numpy as np
from PIL import Image
from scipy import fftpack


def load_image_to_array(path, size=None):
    im = Image.open(path)
    im = im.convert('L')
    if size is not None:
        im = im.resize(size)
    im = np.array(im, dtype=np.float32)
    return im


def array_to_image(arr, clip=False):
    if (np.sum(arr < 0) or np.sum(arr > 255)) and not clip:
        raise Exception('Intensity value out of range.')
    arr = np.clip(arr, 0., 255.)
    arr = arr.astype(np.uint8)
    im = Image.fromarray(arr)
    return im


def add_gaussian_noise(im, sigma):
    im = im.copy()
    im += np.random.normal(loc=0.0, scale=sigma, size=im.shape)
    # im = np.clip(im, 0., 255.)
    return im


def psnr(y, y_hat):
    res = np.mean(np.square(y - y_hat))
    res = 255. * 255. / (res + 10e-8)
    res = 10. * np.log10(res)
    return res


def dct(arr, axes=None, inverse=False):
    if axes is None:
        axes = list(range(len(arr.shape)))
    if isinstance(axes, int):
        axes = [axes]
    for axis in axes:
        if inverse:
            arr = fftpack.idct(arr, axis=axis, norm='ortho')
        else:
            arr = fftpack.dct(arr, axis=axis, norm='ortho')
    return arr


def hard_thr(arr, thr):
    arr = arr.copy()
    arr[np.abs(arr) < thr] = 0.
    return arr


def idx_grid(h, w):
    xv = np.arange(0, w)
    yv = np.arange(0, h)
    xv, yv = np.meshgrid(xv, yv)
    grid = np.stack([yv, xv], -1)
    return grid


if __name__ == '__main__':
    im = load_image_to_array('images/lena.tif')
    print(np.random.rand(256*256, 256*256).shape)