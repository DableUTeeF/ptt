import cv2
import os


def compute_resize_scale(image_shape, min_side=800, max_side=1333):
    """ Compute an image scale such that the image size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resizing scale.
    """
    (rows, cols, _) = image_shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    return scale


def resize_image(img, min_side=800, max_side=1333):
    """ Resize an image such that the size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resized image.
    """
    # compute scale to resize the image
    scale = compute_resize_scale(img.shape, min_side=min_side, max_side=max_side)

    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale


if __name__ == '__main__':
    root = '/media/palm/62C0955EC09538ED/ptt/'
    for folder in os.listdir(os.path.join(root, 'full_sized')):
        if not os.path.isdir(os.path.join(root, 'new_train', folder)):
            os.mkdir(os.path.join(root, 'new_train', folder))
        if not os.path.isdir(os.path.join(root, 'train', folder)):
            os.mkdir(os.path.join(root, 'train', folder))
        if not os.path.isdir(os.path.join(root, 'val', folder)):
            os.mkdir(os.path.join(root, 'val', folder))
        for filename in os.listdir(os.path.join(root, 'full_sized', folder)):
            if filename not in os.listdir(os.path.join(root, 'train', folder))+os.listdir(os.path.join(root, 'val', folder)):
                img = cv2.imread(os.path.join(root, 'full_sized', folder, filename))
                img, _ = resize_image(img, max_side=2000)
                cv2.imwrite(os.path.join(root, 'new_train', folder, filename), img)
