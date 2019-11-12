import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
from skimage.color import rgb2hsv, rgb2gray
from skimage import feature
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.morphology import square, dilation
from skimage.filters import gaussian


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


def shrink(edge, rgb, R1, R2, C1, C2, debug=False):
    # SHRINK
    # rgb = rgb[R1:R2, C1:C2]
    # edge = edge[R1:R2, C1:C2] + bg[R1:R2, C1:C2]
    # edge[edge > 1] = 1
    edge = edge[R1:R2, C1:C2]
    step = 5

    r1, r2, c1, c2 = R1, R2, C1, C2
    R1, R2, C1, C2 = 0, edge.shape[0], 0, edge.shape[1]
    r1_, r2_, c1_, c2_ = R1, R2, C1, C2

    th_area = 40
    ischange = 1
    while ischange > 0:
        # shrink top
        R1_ = R1 + step
        area = np.sum(edge[:R1_, C1:C2])
        while area <= th_area:
            R1 = R1_
            R1_ = R1 + step
            area = np.sum(edge[:R1_, C1:C2])
            if debug:
                print('shrink top', area, R1)

        # shrink bottom
        R2_ = R2 - step
        area = np.sum(edge[R2_:, C1:C2])
        while area <= th_area:
            R2 = R2_
            R2_ = R2 - step
            area = np.sum(edge[R2_:, C1:C2])
            if debug:
                print('shrink bottom', area, R2)

        # shrink left
        C1_ = C1 + step
        area = np.sum(edge[R1:R2, :C1_])
        while area <= th_area:
            C1 = C1_
            C1_ = C1 + step
            area = np.sum(edge[R1:R2, :C1_])
            if debug:
                print('shrink left', area, C1)

        # shrink right
        C2_ = C2 - step
        area = np.sum(edge[R1:R2, C2_:])
        while area <= th_area:
            C2 = C2_
            C2_ = C2 - step
            area = np.sum(edge[R1:R2, C2_:])
            if debug:
                print('shrink right', area, C2)

        ischange = abs(r1_ - R1) + abs(r2_ - R2) + abs(c1_ - C1) + abs(c2_ - C2)
        r1_, r2_, c1_, c2_ = R1, R2, C1, C2

    if debug:
        print('shrink', R1, R2, C1, C2)
        plt.subplot(1, 2, 1)
        plt.imshow(rgb[r1 + R1:r1 + R2, c1 + C1:c1 + C2])
        plt.subplot(1, 2, 2)
        plt.imshow(edge[R1:R2, C1:C2])
        plt.show()

    return R1, R2, C1, C2


# noinspection PyUnboundLocalVariable,PyUnresolvedReferences
def crop(rgb, debug=False):
    # remove black border
    if rgb.dtype == np.uint8:
        black_th = 10
    else:
        black_th = 0.03
    black = (rgb[:, :, 0] < black_th) & (rgb[:, :, 1] < black_th) & (rgb[:, :, 2] < black_th)
    # remove top
    for r1 in range(black.shape[0]):
        if np.sum(black[r1, :]) / black.shape[1] < 0.97:
            break
    if r1 > 0:
        rgb = rgb[r1 + 1:, :, :]
        black = black[r1 + 1:, :]
    # remove bottom
    for r2 in range(black.shape[0] - 1, 0, -1):
        if np.sum(black[r2, :]) / black.shape[1] < 0.97:
            break
    if r2 < black.shape[0] - 1:
        rgb = rgb[:r2, :, :]
        black = black[:r2, :]
    # remove left
    for c1 in range(black.shape[1]):
        if np.sum(black[:, c1]) / black.shape[0] < 0.97:
            break
    if c1 > 0:
        rgb = rgb[:, c1 + 1:, :]
        black = black[:, c1 + 1]
    # remove right
    for c2 in range(black.shape[1] - 1, 0, -1):
        if np.sum(black[:, c2]) / black.shape[0] < 0.97:
            break
    if c2 < black.shape[1] - 1:
        rgb = rgb[:, :c2, :]
        # black = black[:, :c2]

    IM = rgb.copy()
    # resize
    s = 800
    ht, wt = rgb.shape[:2]

    if ht >= wt:
        newsize = (s, int(ht / wt * s))
    else:
        newsize = (int(wt / ht * s), s)

    rgb = resize(rgb, newsize[::-1])
    if debug:
        print('newsize', newsize)
        print('rgb.shape', rgb.shape)

    if debug:
        plt.imshow(rgb)
        plt.show()

    hsv = rgb2hsv(rgb)
    h = hsv[:, :, 0]
    h = gaussian(h, sigma=1)
    if debug:
        print('h.shape', h.shape)
        print('hsv.shape', hsv.shape)
        print('newsize', newsize)
        print('original size', (ht, wt))
        plt.imshow(h)
        plt.show()

    # # remove light
    # white = (rgb[:,:,0] > 0.97) & (rgb[:,:,1] > 0.97) & (rgb[:,:,2] > 0.97)
    # white = dilation(white, square(20))
    # if debug:
    #   plt.imshow(white)
    #   plt.show()
    # L = label(white)
    # max_area = 0
    # i = 0
    # for r in regionprops(L):
    #   if r.area >= max_area:
    #     max_area = r.area
    #     i = r.label
    # white = L == i
    # if debug:
    #   plt.imshow(white)
    #   plt.show()

    area = 50
    tl = np.mean(h[:area, :area])
    tr = np.mean(h[:area, -area:])
    # cl = np.mean(h[h.shape[0]//2-5:h.shape[0]//2+5, :10])
    # cr = np.mean(h[h.shape[0]//2-5:h.shape[0]//2+5, -10:])
    bl = np.mean(h[-area:, :area])
    br = np.mean(h[-area:, -area:])
    if debug:
        print(tl, tr, bl, br)
    # mu = np.mean([tl, tr, bl, br])
    # tl = setmu(tl, mu)
    # tr = setmu(tr, mu)
    # bl = setmu(bl, mu)
    # br = setmu(br, mu)
    # print(tl, tr, bl, br)

    # tol = 0.9
    # ratio = 1
    # while ratio > 0.5:
    #   toh = 2-tol
    #   TL = h[:newsize[1]//2, :newsize[0]//2]
    #   b_TL = (TL > tol*tl) & (TL < toh*tl)
    #   TR = h[:newsize[1]//2, newsize[0]//2:]
    #   b_TR = (TR > tol*tr) & (TR < toh*tr)

    #   # CL = h[newsize[1]//3:newsize[1]//3*2, :newsize[0]//2]
    #   # b_CL = (CL > tol*cl) & (CL < toh*cl)
    #   # CR = h[newsize[1]//3*2:, :newsize[0]//2]
    #   # b_CR = (CR > tol*cr) & (CR < toh*cr)

    #   BL = h[newsize[1]//2:, :newsize[0]//2]
    #   b_BL = (BL > tol*bl) & (BL < toh*bl)
    #   BR = h[newsize[1]//2:, newsize[0]//2:]
    #   b_BR = (BR > tol*br) & (BR < toh*br)
    #   B = np.vstack((np.hstack((b_TL, b_TR)), np.hstack((b_BL, b_BR))))
    #   B = np.invert(B) * 1.0
    #   # mask = np.ones((5, 5))
    #   # B = cv2.erode(B, mask)
    #   # B = cv2.dilate(B, mask)
    #   #B = np.uint8(B * 255)
    #   tol -= 0.1
    #   plt.imshow(B)
    #   plt.show()
    #   ratio = np.sum(B) / np.prod(B.shape)
    #   print(tol)

    tol = 0.9
    ratio = 0
    ratio_th = 0.6
    while ratio < ratio_th:
        toh = 2 - tol
        TL = h[:h.shape[0] // 2, :h.shape[1] // 2]
        b_TL = (TL > tol * tl) & (TL < toh * tl)
        # b_TL[white[:h.shape[0]//2, :h.shape[1]//2]] = True
        ratio = np.sum(b_TL) / np.prod(b_TL.shape)
        tol -= 0.1
        if debug:
            print('tl', ratio)

    tol = 0.9
    ratio = 0
    while ratio < ratio_th:
        toh = 2 - tol
        TR = h[:h.shape[0] // 2, h.shape[1] // 2:]
        b_TR = (TR > tol * tr) & (TR < toh * tr)
        # b_TR[white[:h.shape[0]//2, h.shape[1]//2:]] = True
        ratio = np.sum(b_TR) / np.prod(b_TR.shape)
        tol -= 0.1
        if debug:
            print('tr', ratio)

    tol = 0.9
    ratio = 0
    while ratio < ratio_th:
        toh = 2 - tol
        BL = h[h.shape[0] // 2:, :h.shape[1] // 2]
        b_BL = (BL > tol * bl) & (BL < toh * bl)
        # b_BL[white[h.shape[0]//2:, :h.shape[1]//2]] = True
        ratio = np.sum(b_BL) / np.prod(b_BL.shape)
        tol -= 0.1
        if debug:
            print('bl', ratio)

    tol = 0.9
    ratio = 0
    while ratio < ratio_th:
        toh = 2 - tol
        BR = h[h.shape[0] // 2:, h.shape[1] // 2:]
        b_BR = (BR > tol * br) & (BR < toh * br)
        # b_BR[white[h.shape[0]//2:, h.shape[1]//2:]] = True
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
    if debug:
        print('ratio', ratio)

    # B = clear_border(B)
    label_image = label(B)
    min_area = 50
    bg = np.zeros(B.shape)
    for region in regionprops(label_image):
        if region.area >= min_area and is_near_bound(region.centroid, B.shape):
            bg[label_image == region.label] = 1
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
    # final = rgb[r1:r2, c1:c2]
    # plt.imshow(final)
    # plt.show()

    # gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = rgb2gray(rgb)
    if ratio > 0.2:
        gray = gaussian(gray, sigma=3)
    else:
        gray = gaussian(gray, sigma=2)
    if debug:
        plt.imshow(gray, cmap=plt.cm.gray)
        plt.show()
    sigma = 0.5
    edge = feature.canny(gray, sigma=sigma)
    # edge[white] = False
    edge_step = 0.5
    while np.sum(edge) / np.prod(edge.shape) > 0.01:
        sigma += edge_step
        edge = feature.canny(gray, sigma=sigma)
        # edge[white] = False
    if debug:
        plt.imshow(edge)
        plt.show()
    # edge = erosion(edge, square(1))
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
            # print('top', area)
            # plt.imshow(edge[R1_-step:R1_, C1:C2])
            # plt.show()

        # expand bottom
        R2_ = R2 + step if R2 + step <= rgb.shape[0] else rgb.shape[0]
        area = np.sum(edge[R2_ + 1:R2_ + step, C1:C2])
        while area > 0:
            R2 = R2_
            R2_ = R2 + step if R2 + step <= rgb.shape[0] else rgb.shape[0]
            area = np.sum(edge[R2_ + 1:R2_ + step, C1:C2])
            # print('bottom', area)
            # plt.imshow(edge[R2_+1:R2_+step, C1:C2])
            # plt.show()

        # expand left
        C1_ = C1 - step if C1 - step >= 0 else 0
        area = np.sum(edge[R1:R2, C1_ - step:C1_])
        while area > 0:
            C1 = C1_
            C1_ = C1 - step if C1 - step >= 0 else 0
            area = np.sum(edge[R1:R2, C1_ - step:C1_])
            # print('left', area)
            # plt.imshow(edge[R1:R2, C1_-step:C1_])
            # plt.show()

        # expand right
        C2_ = C2 + step if C2 + step <= rgb.shape[1] else rgb.shape[1]
        area = np.sum(edge[R1:R2, C2_ + 1:C2_ + step])
        while area > 0:
            C2 = C2_
            C2_ = C2 + step if C2 + step <= rgb.shape[1] else rgb.shape[1]
            area = np.sum(edge[R1:R2, C2_ + 1:C2_ + step])
            # print('right', area)
            # plt.imshow(edge[R1:R2, C2_+1:C2_+step])
            # plt.show()

        ischange = abs(r1_ - R1) + abs(r2_ - R2) + abs(c1_ - C1) + abs(c2_ - C2)
        r1_, r2_, c1_, c2_ = R1, R2, C1, C2

        if debug:
            print('expand', R1, R2, C1, C2)
            plt.subplot(1, 2, 1)
            plt.imshow(rgb[R1:R2, C1:C2])
            plt.subplot(1, 2, 2)
            plt.imshow(edge[R1:R2, C1:C2])
            plt.show()

    # R1, R2, C1, C2 = shrink(edge, rgb, R1, R2, C1, C2, debug)

    # SHRINK
    # rgb = rgb[R1:R2, C1:C2]
    # edge = edge[R1:R2, C1:C2] + bg[R1:R2, C1:C2]
    # edge[edge > 1] = 1
    edge = edge[R1:R2, C1:C2]
    step = 5

    r1, r2, c1, c2 = R1, R2, C1, C2
    R1, R2, C1, C2 = 0, edge.shape[0], 0, edge.shape[1]
    r1_, r2_, c1_, c2_ = R1, R2, C1, C2

    th_area = 40
    ischange = 1
    while ischange > 0:
        # shrink top
        R1_ = R1 + step
        area = np.sum(edge[:R1_, C1:C2])
        while area <= th_area:
            R1 = R1_
            R1_ = R1 + step
            area = np.sum(edge[:R1_, C1:C2])
            if debug:
                print('shrink top', area, R1)

        # shrink bottom
        R2_ = R2 - step
        area = np.sum(edge[R2_:, C1:C2])
        while area <= th_area:
            R2 = R2_
            R2_ = R2 - step
            area = np.sum(edge[R2_:, C1:C2])
            if debug:
                print('shrink bottom', area, R2)

        # shrink left
        C1_ = C1 + step
        area = np.sum(edge[R1:R2, :C1_])
        while area <= th_area:
            C1 = C1_
            C1_ = C1 + step
            area = np.sum(edge[R1:R2, :C1_])
            if debug:
                print('shrink left', area, C1)

        # shrink right
        C2_ = C2 - step
        area = np.sum(edge[R1:R2, C2_:])
        while area <= th_area:
            C2 = C2_
            C2_ = C2 - step
            area = np.sum(edge[R1:R2, C2_:])
            if debug:
                print('shrink right', area, C2)

        ischange = abs(r1_ - R1) + abs(r2_ - R2) + abs(c1_ - C1) + abs(c2_ - C2)
        r1_, r2_, c1_, c2_ = R1, R2, C1, C2

    if debug:
        print('shrink', R1, R2, C1, C2)
        plt.subplot(1, 2, 1)
        plt.imshow(rgb[r1 + R1:r1 + R2, c1 + C1:c1 + C2])
        plt.subplot(1, 2, 2)
        plt.imshow(edge[R1:R2, C1:C2])
        plt.show()

    # remove light
    e = edge[R1:R2, C1:C2]
    e = dilation(e, square(20))
    if debug:
        plt.imshow(e)
        plt.show()
    L = label(e)
    reg = regionprops(L)
    if len(reg) > 1:
        RGB = rgb[r1 + R1:r1 + R2, c1 + C1:c1 + C2]
        white = (RGB[:, :, 0] > 0.97) & (RGB[:, :, 1] > 0.97) & (RGB[:, :, 2] > 0.97)
        if np.sum(white) > 0:
            area_ratio = np.zeros(L.max() + 1)
            for REG in reg:
                BG = L == REG.label
                A = BG & white
                area_ratio[REG.label] = np.sum(A) / np.sum(BG)
            label_light = area_ratio.argmax()

            LL = L == label_light
            row, col = np.where(LL)
            rowmin = row.min() - 1
            rowmax = row.max() + 1
            colmin = col.min() - 1
            colmax = col.max() + 1
            D = [0] * 4
            for ROW1 in range(rowmin, -1, -1):
                if np.sum(e[ROW1, :]) > 0:
                    D[0] = abs(ROW1 - rowmin) / L.shape[0]
                    break
            for ROW2 in range(rowmax, e.shape[0]):
                if np.sum(e[ROW2, :]) > 0:
                    D[1] = abs(ROW2 - rowmax) / L.shape[0]
                    break
            for COL1 in range(colmin, -1, -1):
                if np.sum(e[:, COL1]) > 0:
                    D[2] = abs(COL1 - colmin) / L.shape[1]
                    break
            for COL2 in range(colmax, e.shape[1]):
                if np.sum(e[:, COL2]) > 0:
                    D[3] = abs(COL2 - colmax) / L.shape[1]
                    break
            if max(D) > 0.4:
                e = edge[R1:R2, C1:C2] & np.invert(LL)
                ROW, COL = np.where(e)
                R2 = R1 + ROW.max() + 1
                R1 += ROW.min()
                C2 = C1 + COL.max() + 1
                C1 += COL.min()
                # white = dilation(white, square(20))
            if debug:
                print('D', D)
                print('area_ratio', area_ratio)
                plt.imshow(white)
                plt.show()
                plt.imshow(edge[R1:R2, C1:C2])
                # plt.imshow(e[ROW.min():ROW.max(), COL.min():COL.max()])
                plt.title('edge')
                plt.show()
                print('rgb.shape', rgb.shape)
                print('edge.shape', edge.shape)

    gain_row = IM.shape[0] / rgb.shape[0]
    gain_col = IM.shape[1] / rgb.shape[1]
    if debug:
        plt.imshow(IM)
        plt.show()
    rgb = IM

    buffer = 20

    # i1 = max(0, r1+R1-buffer)
    # i2 = min(rgb.shape[0], r1+R2+buffer)
    # j1 = max(0, c1+C1-buffer)
    # j2 = min(rgb.shape[1], c1+C2+buffer)

    i1 = max(0, int((r1 + R1 - buffer) * gain_row))
    i2 = min(rgb.shape[0], int((r1 + R2 + buffer) * gain_row))
    j1 = max(0, int((c1 + C1 - buffer) * gain_col))
    j2 = min(rgb.shape[1], int((c1 + C2 + buffer) * gain_col))

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

    # return np.uint8(final * 255)
    return final


def fn(file, dirpath):
    if 'screenshot' in file.lower():
        return
    src_cls, dest_cls = dirpath.split('|')
    if os.path.exists(os.path.join(dest_cls, file)):
        return
    im = imread(os.path.join(src_cls, file))
    im = crop(im)
    imsave(os.path.join(dest_cls, file), (im).astype('uint8'))


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
            with poolcontext(processes=8) as pool:
                results = pool.map(partial(fn, dirpath=src_cls + '|' + dest_cls), os.listdir(src_cls))
