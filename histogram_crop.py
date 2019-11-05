import cv2
import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    im_path = '/media/palm/New Volume/ptt/100D5300/food_box_12_jian/1/DSC_0619.JPG'
    image = cv2.imread(im_path)[:, :, ::-1]
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 0]
    center_x = int(image.shape[1]/2)
    center_y = int(image.shape[0]/2)
    bin_img = np.zeros((*image.shape[:2],))
    # image = image[:, :, 1]
    mean_color = np.concatenate((hsv[:10, :10], hsv[:10, -10:], hsv[-10:, :10], hsv[-10:, -10:]), axis=0)
    mean_color = np.mean(mean_color)
    # mean_color = np.mean(mean_color, axis=0)
    # for i in range(3):
    # bin_img[:, :] = image[:, :, i] > (mean_color[i]*0.9)
    # bin_img[:, :] = image[:, :, i] < (mean_color[i]*1.1)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    hsv_mask = np.zeros((*image.shape[:2],))
    hsv_mask[:, :] = mean_color*0.9 > hsv
    hsv_mask[:, :] = hsv > mean_color*1.1

    hist_x = np.sum(hsv_mask, axis=0)
    hist_y = np.sum(hsv_mask, axis=1)
    # hist = np.concatenate((h))

    xs = np.where(hist_x > 0)[0]
    ys = np.where(hist_y > 0)[0]

    center = (ys[0]+ys[-1]) // 2, (xs[0]+xs[-1]) // 2

    size = max(abs(xs[0]-xs[-1]), abs(ys[0]-ys[-1])) // 2

    bbox = {'y1': center[0]-size+int(image.shape[0]*0.95), 'y2': center[0]+size+int(image.shape[0]*1.05),
            'x1': center[1]-size+int(image.shape[1]*0.95), 'x2': center[1]+size+int(image.shape[1]*1.05)}

    borderType = cv2.BORDER_REPLICATE

    big_image = cv2.copyMakeBorder(image, image.shape[0], image.shape[0], image.shape[1], image.shape[1], borderType, None, )

    cropped_image = big_image[bbox['y1']:bbox['y2'],
                    bbox['x1']:bbox['x2']
                    ]

    # plt.plot(hist_x)
    # plt.plot(hist_y)
    # plt.figure()
    # plt.imshow(hsv_mask.astype('uint8') * 255)
    # plt.figure()
    # plt.imshow(hsv)
    # plt.figure()
    # plt.imshow(cropped_image)
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    # plt.imshow(hsv_mask.astype('uint8') * 255)
    plt.subplot(2, 2, 2)
    plt.plot(hist_y[::-1], range(len(hist_y)))
    plt.subplot(2, 2, 3)
    plt.plot(hist_x)
    plt.subplot(2, 2, 4)
    plt.imshow(cropped_image)

    plt.show()

    pass
