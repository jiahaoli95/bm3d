import numpy as np
import utils
from scipy.misc import imresize


def all_blocks(im, bsize):
    # Give an image of size H*W
    # return an array of size (H-bsize+1)*(W-bsize+1)*bsize*bsize
    # which are the blocks corresponding to their left-upper corners
    H, W = im.shape
    nb_y = H - bsize + 1
    nb_x = W - bsize + 1
    shifts = []
    for i in range(bsize):
        for j in range(bsize):
            shifts.append(im[i:i+nb_y, j:j+nb_x])
    blocks = np.stack(shifts, -1)
    blocks = np.reshape(blocks, [H-bsize+1, W-bsize+1, bsize, bsize])
    return blocks


def block_dct(im, bsize, inverse=False):
    # Compute dct for all overlapping blocks in an image
    blocks = all_blocks(im, bsize)
    blocks = utils.dct(blocks, axes=[2, 3], inverse=inverse)
    return blocks


def group(ref_block, neighborhood, thr, limit):
    # ref_block is bsize*bsize
    # neighborhood is nsize*nsize*bsize*bsize
    ref_block = ref_block[np.newaxis, np.newaxis, ...]
    d = np.mean(np.square(ref_block-neighborhood), axis=(2, 3))
    mask = d <= thr
    idx_y, idx_x = np.where(mask)
    grp = d[idx_y, idx_x]
    argsort = np.argsort(grp)
    argsort = argsort[:min(len(argsort), limit)]
    idx_x = idx_x[argsort]
    idx_y = idx_y[argsort]
    return idx_x, idx_y


def bm3d(im,
         sigma,
         N1_ht=8,
         N2_ht=16,
         Nstep_ht=6,
         N_S_ht=12,
         lambda2d=0,
         lambda3d=2.7,
         tau_match_ht=2500,
         N1_wie=8,
         N2_wie=16,
         Nstep_wie=5,
         N_S_wie=12,
         tau_match_wie=400):

    blocks_noisy = block_dct(im, bsize=N1_ht)
    bh, bw, _, _ = blocks_noisy.shape
    thr_im_dct = utils.dct(im)
    thr_im_dct = utils.hard_thr(thr_im_dct, lambda2d*sigma)
    thr_im = utils.dct(thr_im_dct, inverse=True)
    blocks_thr = block_dct(thr_im, bsize=N1_ht)
    step1_agg = np.zeros_like(im)
    step1_weights = np.zeros_like(im)
    _ys = set()#debug
    for y in range(0, bh, Nstep_ht):
        for x in range(0, bw, Nstep_ht):
            ref = blocks_thr[y, x]
            Sy_min = max(0, y-N_S_ht)
            Sy_max = min(bh, y+N_S_ht+1)
            Sx_min = max(0, x-N_S_ht)
            Sx_max = min(bw, x+N_S_ht+1)
            neighborhood = blocks_thr[Sy_min:Sy_max, Sx_min:Sx_max]
            idx_x, idx_y = group(ref, neighborhood, tau_match_ht, N2_ht)
            grp = blocks_noisy[Sy_min:Sy_max, Sx_min:Sx_max][idx_y, idx_x]
            grp = utils.dct(grp, axes=0)
            grp = utils.hard_thr(grp, lambda3d*sigma)
            weights = np.square(sigma) * np.sum(grp>0)
            if weights == 0:
                weights = 1.
            weights = 1. / weights
            grp = utils.dct(grp, inverse=True)
            grp *= weights[np.newaxis, ...]
            idx_x = idx_x + Sx_min
            idx_y = idx_y + Sy_min
            for i in range(len(grp)):
                _ys.add((idx_y[i],idx_x[i]))#debug
                step1_agg[idx_y[i]:idx_y[i]+N1_ht, idx_x[i]:idx_x[i]+N1_ht] += grp[i]
                step1_weights[idx_y[i]:idx_y[i]+N1_ht, idx_x[i]:idx_x[i]+N1_ht] += weights
    print(len(_ys))#debug
    basic_im = step1_agg / (step1_weights + 10e-8)

    blocks_noisy = block_dct(im, bsize=N1_wie)
    bh, bw, _, _ = blocks_noisy.shape
    blocks_basic = block_dct(basic_im, bsize=N1_wie)
    step2_agg = np.zeros_like(im)
    step2_weights = np.zeros_like(im)
    for y in range(0, bh, Nstep_wie):
        for x in range(0, bw, Nstep_wie):
            ref = blocks_basic[y, x]
            Sy_min = max(0, y-N_S_wie)
            Sy_max = min(bh, y+N_S_wie+1)
            Sx_min = max(0, x-N_S_wie)
            Sx_max = min(bw, x+N_S_wie+1)
            neighborhood = blocks_basic[Sy_min:Sy_max, Sx_min:Sx_max]
            idx_x, idx_y = group(ref, neighborhood, tau_match_wie, N2_wie)
            mask = np.zeros(neighborhood.shape[:2], dtype=np.bool)
            mask[idx_y, idx_x] = True
            grp_basic = neighborhood[mask]
            n_matched = len(grp_basic)
            grp_basic = grp_basic[:min(n_matched, N2_wie)]
            grp_basic = utils.dct(grp_basic, axes=0)
            W_wie = np.square(grp_basic) / (np.square(grp_basic) + np.square(sigma))
            grp_noisy = blocks_noisy[Sy_min:Sy_max, Sx_min:Sx_max][mask]
            grp_noisy = grp_noisy[:min(n_matched, N2_wie)]
            grp_noisy = utils.dct(grp_noisy, axes=0)
            grp_noisy *= W_wie
            weights = 1. / (np.square(sigma) * np.sum(np.square(W_wie)) + 10e-8)
            grp_noisy = utils.dct(grp_noisy, inverse=True)
            grp_noisy *= weights
            xv = np.arange(Sx_min, Sx_max)
            yv = np.arange(Sy_min, Sy_max)
            xv, yv = np.meshgrid(xv, yv)
            xv = xv[mask]
            xv = xv[:min(n_matched, N2_wie)]
            yv = yv[mask]
            xv = xv[:min(n_matched, N2_wie)]
            for i in range(len(grp_noisy)):
                step2_agg[yv[i]:yv[i]+N1_wie, xv[i]:xv[i]+N1_wie] += grp_noisy[i]
                step2_weights[yv[i]:yv[i]+N1_wie, xv[i]:xv[i]+N1_wie] += weights
    final_im = step2_agg / (step2_weights + 10e-8)

    return basic_im, final_im


if __name__ == '__main__':
    gt_im = utils.load_image_to_array('images/lena.tif', size=[256, 256])
    noisy = utils.add_gaussian_noise(gt_im, 20)
    utils.array_to_image(noisy, clip=True).save('noisy'+'.jpg')
    im = bm3d(noisy, sigma=20)
    basic_im, final_im = im
    print('PSNR for basic estimate is %.1f' % (utils.psnr(gt_im, basic_im)))
    print('PSNR for final estimate is %.1f' % (utils.psnr(gt_im, final_im)))
    utils.array_to_image(basic_im, clip=True).save('bm3d_basic.jpg')
    utils.array_to_image(final_im, clip=True).save('bm3d_final.jpg')
    quit()
    for x in [8, 12, 16, 32]:
        im = bm3d(noisy, sigma=20, N2_ht=x, N2_wie=x)
    # im = utils.dct(gt_im)
    # im = utils.hard_thr(im, 2.7*20)
    # im = utils.dct(im, inverse=True)
    # utils.array_to_image(im, clip=True).save('dct'+'.jpg')
    # quit()
    # grid search to find the best parameters
    N1_ht = [8]
    N2_ht = [16, 32]
    Nstep_ht = [3, 4]
    N_S_ht = [12, 16]
    lambda2d = [2.7]
    lambda3d = [2.7]
    tau_match_ht = [2500]
    N1_wie = [8]
    N2_wie = [16, 32]
    Nstep_wie = [3, 5]
    N_S_wie = [12, 16]
    tau_match_wie = [2500]
    gt_im = utils.load_image_to_array('images/lena.tif', size=[256, 256])
    noisy = utils.add_gaussian_noise(gt_im, 20)
    print('PSNR for noisy image:', utils.psnr(gt_im, noisy))
    for x1 in lambda2d:
        for x2 in lambda3d:
            for x3 in tau_match_ht:
                for x4 in tau_match_wie:
                    # ims = bm3d(noisy, sigma=20, N2_ht=200, N2_wie=200, lambda2d=x1, lambda3d=x2, tau_match_ht=x3, tau_match_wie=x4)
                    ims = bm3d(noisy, sigma=20)
                    for i in range(len(ims)):
                        if i == 0:
                            continue
                        _im = ims[i]
                        print('PSNR for lambda2d=%.1f, lambda3d=%.1f, tau_match_ht=%.1f, tau_match_wie=%.1f is %.1f' % (x1, x2, x3, x4, utils.psnr(gt_im, _im)))
                        utils.array_to_image(_im, clip=True).save('test'+str(np.random.rand())+'.jpg')
    quit()
    # print(np.min(im), np.max(im));quit()
    for i in range(len(ims)):
        _im = ims[i]
        _im = np.clip(_im, 0, 255.)
        print(utils.psnr(gt_im, _im))
        utils.array_to_image(_im).save('test%d.jpg'%i)
    pass