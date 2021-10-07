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
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, ndimage # requires scipy version 1.4.1 to operate GPyOpt version 1.2.6


def segment_on_dt(a, img, threshold):
    '''
    Implements watershed segmentation.

    Inputs:
    a         := the raw image input
    img       := threshold binned image
    threshold := RGB threshold value

    Outputs:
    lbl       := Borders of segmented droplets
    wat       := Segmented droplets via watershed
    lab       := Indexes of each segmented droplet
    '''
    # estimate the borders of droplets based on known and unknown background + foreground (computed using dilated and erode)
    border = cv2.dilate(img, None, iterations=1)
    border = border - cv2.erode(border, None)
    # segment droplets via distance mapping and thresholding
    dt = cv2.distanceTransform(img, 2, 3)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
    _, dt = cv2.threshold(dt, threshold, 255, cv2.THRESH_BINARY)
    # obtain the map of segmented droplets with corresponding indices
    lbl, ncc = ndimage.label(dt)
    lbl = lbl * (255 / (ncc + 1))
    lab = lbl
    # Completing the markers now.
    lbl[border == 255] = 255
    lbl = lbl.astype(np.int32)
    a = cv2.cvtColor(a,
                     cv2.COLOR_GRAY2BGR)  # we must convert grayscale to BGR because watershed only accepts 3-channel inputs
    wat = cv2.watershed(a, lbl)
    lbl[lbl == -1] = 0
    lbl = lbl.astype(np.uint8)
    return 255 - lbl, wat, lab  # return lab, the segmented and indexed droplets


def watershed_segment(image, double_watershed, large_elements_pixels, pixel_diff, drop_dilate, plot_pixel_diff,
                      remove_artefacting):
    '''
    Applies one- or two-fold watershed image segmentation to separate droplet pixels from background pixels.

    Inputs:
    image                   := input droplet image to segment
    double_watershed        := True or False value. Determines whether to do single or double watershed.
                               Generally, do True if droplets are big, False is droplets are small.
                               IF FALSE, ARGUMENT VALUES OF "large_elements_pixels", "small_elements_pixels", and "drop_dilate" DON'T MATTER.
    large_elements_pixels   := Cleans large elements that contain more than specified number of pixels.
                               Helpful for removing artefacting borders.
    pixel_diff              := Difference of the number of pixels between regular droplets and small elements.
                               Small elements will be removed if the calculated difference >= user-defined pixel_diff, otherwise pass.
                               Helpful for removing spaces between droplets.
    drop_dialate            := Determines how much to fill in the white interior droplets with black.
    plot_pixel_diff         := True or False value. Plots the segmentation figure of small elements from droplets.
    remove_artefacting      := True or False value. Uses scipy signal processing to remove artefacts of spaces between
                               droplets being segmented as actual droplets. Enabling this may accidentally remove real droplets.

    Outputs:
    droplet_count           := Image of droplet interiors indexed by droplet number
    binarized               := Binary image indicating total droplet area vs. empty tube space
    '''
    RGB_threshold = 0
    pixel_threshold = 0
    img = image.copy()

    if double_watershed == False:
        img = 255 - img
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.medianBlur(img, 5)
    _, img_bin = cv2.threshold(img, 0, 255,
                               # threshold image using Otsu's binarization # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
                               cv2.THRESH_OTSU)
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN,
                               np.ones((4, 4), dtype=int))
    # first fold of watershed to remove white centers
    result, water, labs = segment_on_dt(a=img, img=img_bin,
                                        threshold=RGB_threshold)  # segment droplets from background and return indexed droplets

    if double_watershed == True:
        # remove large elements
        large_elements_thresh = large_elements_pixels
        uniq_full, uniq_counts = np.unique(water,
                                           return_counts=True)  # get all unique watershed indices with pixel counts
        large_elements = uniq_full[uniq_counts > large_elements_thresh]  # mask large elements
        for n in range(len(large_elements)):
            water[water == large_elements[n]] = 0  # remove all large elements
        uniq_full, uniq_counts = np.unique(water,
                                           return_counts=True)  # update list of unique watershed indixes and pixel counts
        uniq_vis_y = np.sort(uniq_counts[1:])  # sort the remaining counts from smallest to largest
        uniq_delta = []  # initialize list to populate
        Y = 5  # number of elements to take the difference between: n and n+Y elements
        # remove small elements
        for n in range(len(uniq_vis_y) - Y):
            uniq_delta.append(uniq_vis_y[n + Y] - uniq_vis_y[n])  # take the difference between n and n+Y elements
        uniq_delta = np.array(uniq_delta)  # convert to np array
        small_elements_thresh = uniq_vis_y[np.argmax(uniq_delta)]  # find index where n and n+Y difference is largest

        if plot_pixel_diff == True:
            uniq_vis_x = np.arange(0, len(uniq_vis_y), 1)
            plt.figure(figsize=(4, 4))
            plt.plot(uniq_vis_x, uniq_vis_y)
            plt.axvline(np.argmax(uniq_delta), color='r', linestyle='--')
            plt.title('First-fold watershed segmentation\nof small elements from droplets')
            plt.xlabel('Element index')
            plt.ylabel('# of pixels')
            plt.show()
        else:
            pass
        if small_elements_thresh >= pixel_diff:  # only if calculated threshold is larger than the user defined small elements do we remove small elements
            small_elements = uniq_full[uniq_counts <= small_elements_thresh]  # mask small elements
            for n in range(len(small_elements)):
                water[water == small_elements[n]] = 0  # remove all small elements
        else:
            pass

        #        Remove artefacting spaces between droplets, only necessary for double watershed
        #        Remove artefacting removes the erroneously segmented spaces between droplets as actual droplets. Enabling this may remove actual droplets by accident.
        if remove_artefacting:
            uniq_cleaned = np.unique(water)  # update list of unique indices after cleaning
            for n in range(len(uniq_cleaned)):
                shapetest = water.copy()  # reset each iteration
                shapetest[shapetest != uniq_cleaned[n]] = 0
                if uniq_cleaned[n] > 0:
                    shapetest = shapetest / uniq_cleaned[n]
                else:
                    pass
                chord_v = np.sum(shapetest, axis=0)  # find the sum of object pixels along x-axis
                chord_h = np.sum(shapetest, axis=1)  # find the sum of object pizels along the y-axis
                diff_v = signal.find_peaks(chord_v)  # find the peaks of data
                diff_h = signal.find_peaks(chord_h)  # find the peaks of data
                if (len(diff_v[0]) > 1 or len(diff_h[0]) > 1):  # if the data is polymodal, remove the objects
                    water[water == uniq_cleaned[n]] = 0
                else:
                    pass
        # second fold of watershed using cleaned image as a base
        water = water / 1.
        kernel = np.ones((drop_dilate, drop_dilate), np.uint8)
        water1 = cv2.dilate(water, kernel, iterations=1)
        base = image.copy()  # begin filling in the droplet centers
        base = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)  # convert image to greyscale
        centers = np.where(water1 != 0)  # select all watershed segmented centers
        base[centers[0], centers[1]] = np.average(
            np.unique(base)) / 4  # convert the color of droplet centers to the darkest color of the base image
        base1 = 255 - base
        _, img_bin = cv2.threshold(base1, 0, 255,
                                   # threshold image using Otsu's binarization # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
                                   cv2.THRESH_OTSU)
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN,
                                   np.ones((5, 5), dtype=int))
        result_double, water_double, labs_double = segment_on_dt(a=base, img=img_bin,
                                                                 threshold=RGB_threshold)  # segment droplets from background and return indexed droplets
        result_double[result_double == 255] = 0
        droplet_count = water1.copy()
        return droplet_count

    elif double_watershed == False:
        result[result == 255] = 0
        droplet_count = result.copy()
        return droplet_count
    else:
        raise ValueError("Argument 'double_watershed' takes value either True or False.")