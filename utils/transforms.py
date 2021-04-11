"""
Affine transforms implemented on torch tensors, and
requiring only one interpolation
"""

import random
import cv2
import numpy as np


def th_affine2d(x, matrix, output_img_width, output_img_height):
    """
    2D Affine image transform on torch.Tensor

    """
    assert matrix.ndim == 2
    matrix = matrix[:2, :]
    transform_matrix = matrix
    src = x

    # cols, rows, channels = src.shape
    dst = cv2.warpAffine(
        src,
        transform_matrix,
        (output_img_width, output_img_height),
        cv2.INTER_AREA,
        cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    # for gray image
    if dst.ndim == 2:
        dst = np.expand_dims(np.asarray(dst), axis=2)

    return dst


def initAlignTransfer(img_size, crop_size):
    transform_align = AffineCompose(
        random_crop_size=crop_size,
        rotation_range=7.5,
        translation_range=7.5,
        zoom_range=[1.075, 1.15],
        fine_size=img_size,
        mirror=True,
    )
    return transform_align


# input img: numpy, 3*256*256, 0-255


class AffineCompose(object):
    def __init__(
        self,
        random_crop_size,
        rotation_range,
        translation_range,
        zoom_range,
        fine_size,
        mirror,
    ):

        self.fine_size = fine_size
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.zoom_range = zoom_range
        self.mirror = mirror
        self.random_crop_size = random_crop_size

    def __call__(self, *input):
        rotate = random.uniform(-self.rotation_range, self.rotation_range)
        trans_x = random.uniform(-self.translation_range, self.translation_range)
        trans_y = random.uniform(-self.translation_range, self.translation_range)
        if not isinstance(self.zoom_range, list) and not isinstance(
            self.zoom_range, tuple
        ):
            raise ValueError("zoom_range must be tuple or list with 2 values")
        zoom = random.uniform(self.zoom_range[0], self.zoom_range[1])

        # rotate
        transform_matrix = np.zeros([3, 3])
        center = (self.fine_size / 2.0 - 0.5, self.fine_size / 2 - 0.5)
        M = cv2.getRotationMatrix2D(center, rotate, 1)
        transform_matrix[:2, :] = M
        transform_matrix[2, :] = np.array([[0, 0, 1]])
        # translate
        transform_matrix[0, 2] += trans_x
        transform_matrix[1, 2] += trans_y
        # zoom
        for i in range(3):
            transform_matrix[0, i] *= zoom
            transform_matrix[1, i] *= zoom
        transform_matrix[0, 2] += (1.0 - zoom) * center[0]
        transform_matrix[1, 2] += (1.0 - zoom) * center[1]

        # mirror about x axis in cropped image
        do_mirror = False
        if self.mirror:
            mirror_rng = random.uniform(0.0, 1.0)
            if mirror_rng > 0.5:
                do_mirror = True
        if do_mirror:
            transform_matrix[0, 0] = -transform_matrix[0, 0]
            transform_matrix[0, 1] = -transform_matrix[0, 1]
            transform_matrix[0, 2] = float(self.fine_size) - transform_matrix[0, 2]

        input_tf = input[0]

        # all other transforms
        input_tf = th_affine2d(
            input_tf,
            transform_matrix,
            output_img_width=self.fine_size,
            output_img_height=self.fine_size,
        )

        # # random crop
        # if self.random_crop_size != None:
        #     if self.fine_size > self.random_crop_size:
        #         x1 = random.randint(0, self.fine_size - self.random_crop_size)
        #         y1 = random.randint(0, self.fine_size - self.random_crop_size)
        #         input_tf = input_tf[x1:x1 + self.random_crop_size, y1:y1 + self.random_crop_size]
        #         # input_tf = cv2.resize(input_tf, (self.fine_size, self.fine_size))

        return input_tf
