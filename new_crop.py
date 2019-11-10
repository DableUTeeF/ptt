import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import rgb2hsv, rgb2gray
from skimage import feature
from skimage.io import imread, imsave
from skimage.transform import resize


def setmu(x, mu):
    return mu if abs(mu - x) / mu > 0.1 else x


def is_near_bound(pt, sz, near=0.1):
    tp = sz[0] * near
    bt = sz[0] * (1 - near)
    lf = sz[1] * near
    rt = sz[1] * (1 - near)
    if pt[1] > lf and pt[1] < rt and pt[0] < bt and pt[0] > tp:
        return True
    return False


def crop(rgb, debug=False):
    # rgb = imread(fn)
    s = 800
    ht, wt = rgb.shape[:2]
    if ht >= wt:
        newsize = (s, int(ht / wt * s))
    else:
        newsize = (int(wt / ht * s), s)
    rgb = resize(rgb, newsize[::-1])
    hsv = rgb2hsv(rgb)
    h = hsv[:, :, 0]
    if debug:
        plt.imshow(h)
        plt.show()

    # remove light
    white = (rgb[:, :, 0] > 0.97) & (rgb[:, :, 1] > 0.97) & (rgb[:, :, 2] > 0.97)
    white = closing(white, square(10))
    h[white] = np.median(h)
    if debug:
        plt.imshow(white)
        plt.show()
        plt.imshow(h)
        plt.show()

    area = 50
    tl = np.mean(h[:area, :area])
    tr = np.mean(h[:area, -area:])
    # cl = np.mean(h[h.shape[0]//2-5:h.shape[0]//2+5, :10])
    # cr = np.mean(h[h.shape[0]//2-5:h.shape[0]//2+5, -10:])
    bl = np.mean(h[-area:, :area])
    br = np.mean(h[-area:, -area:])
    if debug:
        print(tl, tr, bl, br)
    tol = 0.9
    ratio = 0
    while ratio < 0.6:
        toh = 2 - tol
        TL = h[:newsize[1] // 2, :newsize[0] // 2]
        b_TL = (TL > tol * tl) & (TL < toh * tl)
        ratio = np.sum(b_TL) / np.prod(b_TL.shape)
        tol -= 0.1
        if debug:
            print('tl', ratio)

    tol = 0.9
    ratio = 0
    while ratio < 0.6:
        toh = 2 - tol
        TR = h[:newsize[1] // 2, newsize[0] // 2:]
        b_TR = (TR > tol * tr) & (TR < toh * tr)
        ratio = np.sum(b_TR) / np.prod(b_TR.shape)
        tol -= 0.1
        if debug:
            print('tr', ratio)

    tol = 0.9
    ratio = 0
    while ratio < 0.6:
        toh = 2 - tol
        BL = h[newsize[1] // 2:, :newsize[0] // 2]
        b_BL = (BL > tol * bl) & (BL < toh * bl)
        ratio = np.sum(b_BL) / np.prod(b_BL.shape)
        tol -= 0.1
        if debug:
            print('bl', ratio)

    tol = 0.9
    ratio = 0
    while ratio < 0.6:
        toh = 2 - tol
        BR = h[newsize[1] // 2:, newsize[0] // 2:]
        b_BR = (BR > tol * br) & (BR < toh * br)
        ratio = np.sum(b_BR) / np.prod(b_BR.shape)
        tol -= 0.1
        if debug:
            print('br', ratio)

    B = np.vstack((np.hstack((b_TL, b_TR)), np.hstack((b_BL, b_BR))))
    B = np.invert(B) * 1.0

    if debug:
        plt.imshow(B)
        plt.show()

    ratio = np.sum(B) / np.prod(B.shape)

    # B = clear_border(B)
    label_image = label(B)
    min_area = 50
    bg = np.zeros(B.shape)
    for region in regionprops(label_image):
        if region.area >= min_area and is_near_bound(region.centroid, B.shape):
            bg[label_image == region.label] = 1
            region.centroid
    if debug:
        plt.imshow(bg)
        plt.show()

    hist_x = np.sum(bg, axis=0)
    hist_y = np.sum(bg, axis=1)
    if debug:
        plt.subplot(1, 2, 1)
        plt.plot(hist_x)
        plt.subplot(1, 2, 2)
        plt.plot(hist_y)
        plt.show()

    idx_x = np.where(hist_x > 0)[0]
    idx_y = np.where(hist_y > 0)[0]
    buffer = 40
    r1 = idx_y[0] - buffer if idx_y[0] - buffer >= 0 else idx_y[0]
    r2 = idx_y[-1] + buffer if idx_y[-1] + buffer < rgb.shape[0] else idx_y[-1]
    c1 = idx_x[0] - buffer if idx_x[0] - buffer >= 0 else idx_x[0]
    c2 = idx_x[-1] + buffer if idx_x[-1] + buffer < rgb.shape[1] else idx_x[-1]
    gray = rgb2gray(rgb)
    if debug:
        plt.imshow(gray, cmap=plt.cm.gray)
        plt.show()
    sigma = 0.5
    edge = feature.canny(gray, sigma=sigma)
    edge_step = 0.5
    while np.sum(edge) / np.prod(edge.shape) > 0.01:
        sigma += edge_step
        edge = feature.canny(gray, sigma=sigma)
    if debug:
        plt.imshow(edge)
        plt.show()
    # edge = closing(edge, square(5))
    # plt.imshow(edge)
    # plt.show()

    # EXPAND
    step = 5

    R1, R2, C1, C2 = r1, r2, c1, c2
    r1_, r2_, c1_, c2_ = r1, r2, c1, c2

    ischange = 1
    while ischange > 0:
        # expand top
        R1_ = R1 - step if R1 - step >= 0 else 0
        area = np.sum(edge[R1_ - step:R1_, C1:C2])
        while area > 0:
            R1 = R1_
            R1_ = R1 - step if R1 - step >= 0 else 0
            area = np.sum(edge[R1_ - step:R1_, C1:C2])

        R2_ = R2 + step if R2 + step <= rgb.shape[0] else rgb.shape[0]
        area = np.sum(edge[R2_ + 1:R2_ + step, C1:C2])
        while area > 0:
            R2 = R2_
            R2_ = R2 + step if R2 + step <= rgb.shape[0] else rgb.shape[0]
            area = np.sum(edge[R2_ + 1:R2_ + step, C1:C2])

        # expand left
        C1_ = C1 - step if C1 - step >= 0 else 0
        area = np.sum(edge[R1:R2, C1_ - step:C1_])
        while area > 0:
            C1 = C1_
            C1_ = C1 - step if C1 - step >= 0 else 0
            area = np.sum(edge[R1:R2, C1_ - step:C1_])

        # expand right
        C2_ = C2 + step if C2 + step <= rgb.shape[1] else rgb.shape[1]
        area = np.sum(edge[R1:R2, C2_ + 1:C2_ + step])
        while area > 0:
            C2 = C2_
            C2_ = C2 + step if C2 + step <= rgb.shape[1] else rgb.shape[1]
            area = np.sum(edge[R1:R2, C2_ + 1:C2_ + step])

        ischange = abs(r1_ - R1) + abs(r2_ - R2) + abs(c1_ - C1) + abs(c2_ - C2)
        r1_, r2_, c1_, c2_ = R1, R2, C1, C2

        if debug:
            print('expand', R1, R2, C1, C2)
            plt.subplot(1, 2, 1)
            plt.imshow(rgb[R1:R2, C1:C2])
            plt.subplot(1, 2, 2)
            plt.imshow(edge[R1:R2, C1:C2])
            plt.show()

    # SHRINK
    # rgb = rgb[R1:R2, C1:C2]
    edge = edge[R1:R2, C1:C2]
    step = 5

    r1, r2, c1, c2 = R1, R2, C1, C2
    R1, R2, C1, C2 = 0, edge.shape[0], 0, edge.shape[1]
    r1_, r2_, c1_, c2_ = R1, R2, C1, C2

    th_area = 50
    ischange = 1
    while ischange > 0:
        # shrink top
        R1_ = R1 + step
        area = np.sum(edge[:R1_, C1:C2])
        while area <= th_area:
            R1 = R1_
            R1_ = R1 + step
            area = np.sum(edge[:R1_, C1:C2])
            if debug: print('shrink top', area)

        # shrink bottom
        R2_ = R2 - step
        area = np.sum(edge[R2_:, C1:C2])
        while area <= th_area:
            R2 = R2_
            R2_ = R2 - step
            area = np.sum(edge[R2_:, C1:C2])
            if debug: print('shrink bottom', area)

        # shrink left
        C1_ = C1 + step
        area = np.sum(edge[R1:R2, :C1_])
        while area <= th_area:
            C1 = C1_
            C1_ = C1 + step
            area = np.sum(edge[R1:R2, :C1_])
            if debug: print('shrink left', area)

        # shrink right
        C2_ = C2 - step
        area = np.sum(edge[R1:R2, C2_:])
        while area <= th_area:
            C2 = C2_
            C2_ = C2 - step
            area = np.sum(edge[R1:R2, C2_:])
            if debug: print('shrink right', area)

        ischange = abs(r1_ - R1) + abs(r2_ - R2) + abs(c1_ - C1) + abs(c2_ - C2)
        r1_, r2_, c1_, c2_ = R1, R2, C1, C2

        if debug:
            print('shrink', R1, R2, C1, C2)
            plt.subplot(1, 2, 1)
            plt.imshow(rgb[r1 + R1:r1 + R2, c1 + C1:c1 + C2])
            plt.subplot(1, 2, 2)
            plt.imshow(edge[R1:R2, C1:C2])
            plt.show()

    buffer = 10
    i1 = max(0, r1 + R1 - buffer)
    i2 = min(rgb.shape[0], r1 + R2 + buffer)
    j1 = max(0, c1 + C1 - buffer)
    j2 = min(rgb.shape[1], c1 + C2 + buffer)
    ar = abs(i1 - i2)
    ac = abs(j1 - j2)
    if ar > ac:
        center = (j1 + j2) // 2
        J1 = max(0, center - (ar // 2))
        J2 = min(rgb.shape[1], center + (ar // 2))
        final = rgb[i1:i2, J1:J2]
    else:
        center = (i1 + i2) // 2
        I1 = max(0, center - (ac // 2))
        I2 = min(rgb.shape[0], center + (ac // 2))
        final = rgb[I1:I2, j1:j2]

    diff = final.shape[0] - final.shape[1]
    if diff > 0:
        padL = final[:, 0, :][:, None, :]
        padR = final[:, -1, :][:, None, :]
        for rep in range(diff // 2):
            final = np.hstack((padL, final, padR))
        if diff % 2 != 0:
            final = np.hstack((final, padR))
    elif diff < 0:
        padT = final[0, :, :][None, :, :]
        padB = final[-1, :, :][None, :, :]
        for rep in range(-diff // 2):
            final = np.vstack((padT, final, padB))
        if -diff % 2 != 0:
            final = np.vstack((final, padB))

    if debug:
        print('final', R1, R2, C1, C2)
        plt.imshow(final)
        plt.show()
        print(final.shape)

    return final


def fn(file, dirpath):
    if 'screenshot' in file.lower():
        return
    if os.path.exists(os.path.join(dirpath, file)):
        return
    src_cls, dest_cls = dirpath.split('|')
    im = imread(os.path.join(src_cls, file))
    im = crop(im)
    imsave(os.path.join(dest_cls, file), (im * 255).astype('uint8'))


if __name__ == '__main__':
    import os
    import multiprocessing
    from functools import partial
    from contextlib import contextmanager


    @contextmanager
    def poolcontext(*args, **kwargs):
        pool = multiprocessing.Pool(*args, **kwargs)
        yield pool
        pool.terminate()


    root = '/media/palm/62C0955EC09538ED/ptt'
    src = ['train', 'val']
    dest = ['new_train', 'new_val']
    for i in range(2):
        for cls in os.listdir(os.path.join(root, src[i])):
            dest_cls = os.path.join(root, dest[i], cls)
            src_cls = os.path.join(root, src[i], cls)
            if not os.path.isdir(os.path.join(root, dest[i], cls)):
                os.mkdir(os.path.join(root, dest[i], cls))
            with poolcontext(processes=3) as pool:
                results = pool.map(partial(fn, dirpath=src_cls + '|' + dest_cls), os.listdir(src_cls))
