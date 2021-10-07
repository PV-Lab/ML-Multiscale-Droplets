# Copyright (c) 2021 Alexander E. Siemenn, Iddo Drori, Matthew J. Beveridge
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of
# conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or other materials provided with the
# distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import cv2 # pip install opencv-python          # https://pypi.org/project/opencv-python/
from scipy import ndimage

def read_rotate_crop(img_path, rotate_crop_params):
    '''
    Rotates and crops the given image.

    Inputs:
    img                  := image path
    rotate_crop_params   := dictionary of values: {theta, x1, x2, y1, y2}, where
        theta            := angle of counter clockwise rotation
        x1               := start pixel of x-axis crop
        x2               := end pixel of x-axis crop
        y1               := start pixel of y-axis crop
        y2               := end pixel of y-axis crop

    Ouputs:
    img                  := rotated and cropped image
    '''
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # read images
    rotated = ndimage.rotate(img, rotate_crop_params['theta'])  # reads image and rotates
    img = rotated[rotate_crop_params['y1']:rotate_crop_params['y2'],
          rotate_crop_params['x1']:rotate_crop_params['x2']]  # crops image
    return img