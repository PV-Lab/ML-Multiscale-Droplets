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

def yield_loss(droplet_count, max_droplets):
    ''''
    Calculates yield loss => the balanced optimization of having many droplets with large area footprint.
    If droplets are both large and there are many of them, they obtain a low yield loss. We aim to minimize yield loss.

    Inputs:
    droplet_count       := watershed segmented indexed droplets.
    max_droplets        := A user defined estimate of the max number of droplets per any given image.
                           This is used to keep yield loss as a minimization problem.

    Outputs:
    yield_loss     := value in range[0,1] that represents how closely all droplets are to having both large area and high count.
    '''
    max_droplets = max_droplets * 1.5 # make max_droplets sufficiently large to keep yield loss < 1
    num_droplets = len(np.unique(droplet_count))  # gets number of unique droplets
    count_loss = (max_droplets - num_droplets) / max_droplets  # pick max number of droplets to create a loss value (i.e., we want to get as close as possible to the max number of droplets) then normalize. This max number of droplets should be larger than the expected achieveable value so it is always <= 1.0
    binarized = droplet_count.copy()
    binarized[binarized != 0] = 1 # assign all droplets to index 1 and background to index 0
    area_loss = np.count_nonzero(binarized == 0) / binarized.size  # background pixels/total pixels ==> want to minimize this
    yld_loss = (count_loss + area_loss) / 2 # minimize this yield loss. We want to maximize number of droplets but also maximize the area of each droplet.
    return yld_loss

def geometric_loss(droplet_count, image_name, iter_plot):
    '''
     Calculates geometric loss => how closely each droplet maps to a perfect circle using computer vision.
     If droplets are close to the circle, they obtain a low loss score.

     Inputs:
     droplet_count              := watershed segmented droplet count image, each individual droplet should have uniquely indexed pixels
     image_name                 := Name of droplet image used to organize to the droplet geometric data list.
     iter_plot                  := True or False value determines whether or not to iteratively plot each droplet mapping to a perfect circle

     Outputs:
     geometric_loss             := value in range[0,1] that represents how closely ALL droplets map to a perfect circle
     droplet_geometry           := list of lists, where each sublist corresponds to the geometric properties of a single droplet.
                                   This variable has the format: ['image name','droplet number', 'centroid_x position','centroid_y position','chord_x length','chord_y length','number of pixels']
     '''
    loss_list = []
    uniq_select = np.unique(droplet_count)
    droplet_geometry = []
    for n in range(len(uniq_select)):
        if n == 0: # pass on background
            pass
        else:
            uniq = uniq_select[n]
            drops = droplet_count.copy()
            drops[drops != uniq] = 0
            drops[drops == uniq] = 1
            circle_init = np.zeros(np.shape(drops))

            # axis 1, summing each row
            uniq_ax1 = np.sum(drops, axis=1)
            start_ax1 = np.where(uniq_ax1 == uniq_ax1[uniq_ax1 > 0][0])[0][0]  # get the index of the first non-zero row value
            mid_ax1 = start_ax1 + sum(np.sum(drops, axis=1) > 0) // 2  # sum only true values to find the length
            diam_ax1 = (np.sum(drops, axis=1)[mid_ax1] + np.max(uniq_ax1)) / 2

            # axis 0, summing each column
            uniq_ax0 = np.sum(drops, axis=0)
            start_ax0 = np.where(uniq_ax0 == uniq_ax0[uniq_ax0 > 0][0])[0][0]  # get the index of the first non-zero row value
            mid_ax0 = start_ax0 + sum(np.sum(drops, axis=0) > 0) // 2  # sum only true values to find the length
            diam_ax0 = (np.sum(drops, axis=0)[mid_ax0] + np.max(uniq_ax0)) / 2

            radi = int((diam_ax0 / 2 + diam_ax1 / 2) / 2)

            circ = cv2.circle(circle_init, (mid_ax0, mid_ax1), radi, (1, 1, 1), -1)
            inside = drops + circ - 1
            inside[inside < 0] = 0
            total = np.abs(circ - drops) + inside

            lossl = (np.sum(np.abs(circ - drops)) / np.sum(total)) * np.sum(drops)  # fraction of droplet not matching perfect circle weighted by the number of pixels in that droplet
            loss_list.append(lossl)

            # append geoemtry data to dataframe for every droplet in every sample
            droplet_geometry.append([image_name, n, mid_ax0, mid_ax1, diam_ax0, diam_ax1, np.sum(drops)])  # ['image name','droplet number', 'centroid_x position','centroid_y position','chord_x length','chord_y length','number of pixels']

            if iter_plot:
                fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
                x0 = int(mid_ax0 - np.max([radi, np.max(uniq_ax0)])) - 20
                x1 = int(mid_ax0 + np.max([radi, np.max(uniq_ax0)])) + 20
                y0 = int(mid_ax1 - np.max([radi, np.max(uniq_ax1)])) - 20
                y1 = int(mid_ax1 + np.max([radi, np.max(uniq_ax1)])) + 20
                ax1.imshow(drops)
                ax1.scatter(mid_ax0, mid_ax1, c='r', s=1)
                ax1.set_title('Imaged Droplet #' + str(n))
                ax1.set_xlim([x0, x1])  # zoom into droplet
                ax1.set_ylim([y0, y1])  # zoom into droplet
                ax2.imshow(circ)  # zoom into droplet
                ax2.scatter(mid_ax0, mid_ax1, c='r', s=1)
                ax2.set_title('Estimated Droplet\nLoss Label = ' + str(round(lossl, 3)))
                ax2.set_xlim([x0, x1])  # zoom into droplet
                ax2.set_ylim([y0, y1])  # zoom into droplet
                plt.show()

    total_pixels = np.sum(droplet_count != 0)  # total number of droplet pixels
    geom_loss = np.sum(loss_list) / total_pixels  # sum the individual losses (each weighted by # of pixels in each droplet) and divide by total number of droplet pixels
    return geom_loss, droplet_geometry