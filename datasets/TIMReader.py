import numpy as np
from PIL import Image
from pylab import *
import math
from scipy.stats import multivariate_normal
import os

TEST_ROOT = "/home/suidong/Documents/TMI2015/testing/"
TRAIN_ROOT = "/home/suidong/Documents/output/tim2015/learning/"
# TEST_ROOT = "/Users/karezi/Desktop/TIM2015/testing/"
# TRAIN_ROOT = "/Users/karezi/Desktop/TIM2015/learning/"
IM_W = 128
IM_H = 128
SZ = 27  # 7
# SZ *= 5
HSZ = SZ//2
COVSZ = 20  # 10
# COVSZ *= 5
COEF = 0.159  # 0.159


class TIMReader(object):
    pad = None

    def __init__(self):
        self.pad = self.create_labelpad()
        for i in range(516):
            self.tif_border_files = Image.open(TEST_ROOT + "%03d_block.tif" % (i + 1))
            self.tif_dot_files = Image.open(TEST_ROOT + "%03d_cell.tif" % (i + 1))
            self.tif_img_files = Image.open(TEST_ROOT + "%03d.tif" % (i + 1))
            self.show_info(self.tif_img_files)
            border = self.get_border(self.tif_border_files)
            im_set = self.split(self.tif_img_files, border)
            dot_set = self.split(self.tif_dot_files, border)
            map_set = self.gen_map(dot_set)
            # self.show_figure(1, im_set)
            # self.show_figure(2, dot_set)
            # self.show_figure(3, map_set)
            self.save_set(im_set, TRAIN_ROOT + "images", (i + 1))
            self.save_set(map_set, TRAIN_ROOT + "maps", (i + 1))

    def gen_map(self, dot_set):
        map_set = []
        for (i, label_im) in enumerate(dot_set):
            coordinates = self.get_coordinates(label_im)
            w, h = label_im.size
            label_arr = np.zeros([w, h, 1])
            for x, y in coordinates:
                x = int(x + 2)
                y = int(y + 2)
                # if x - HSZ >= 0 and x + HSZ + 1 <= w and y - HSZ >= 0 and y + HSZ + 1 <= h:
                #     label_arr[x - HSZ:x + HSZ + 1, y - HSZ:y + HSZ + 1, 0] += self.create_labelpad()
                if x - HSZ < 0:
                    a = 0
                    al = HSZ - x
                else:
                    a = x - HSZ
                    al = 0
                if x + HSZ + 1 > w:
                    b = w
                    bl = SZ - (x + HSZ + 1 - w)
                else:
                    b = x + HSZ + 1
                    bl = SZ
                if y - HSZ < 0:
                    c = 0
                    cl = HSZ - y
                else:
                    c = y - HSZ
                    cl = 0
                if y + HSZ + 1 > h:
                    d = h
                    dl = SZ - (y + HSZ + 1 - h)
                else:
                    d = y + HSZ + 1
                    dl = SZ
                # print(a, b, c, d)
                # print(al, bl, cl, dl)
                label_arr[a:b, c:d, 0] += self.pad[al:bl, cl:dl]

            # if numpy.sum(label_arr) < 100:
            #     continue
            im = label_arr[..., 0].T
            im = Image.fromarray(np.array(im * 255).astype('uint8'))
            map_set.append(im)
        # print(map_set.__len__())
        return map_set

    @staticmethod
    def save_set(im_set, cat, prefix_no):
        for i, im in enumerate(im_set):
            im.save(os.path.join(cat, "%03d.%d.png" % (prefix_no, i + 1)), 'PNG')

    @staticmethod
    def show_figure(no, set):
        plt.figure(no)
        for i in range(9):
            plt.subplot(331 + i)
            plt.imshow(set[i])
        plt.show()

    @staticmethod
    def show_info(im):
        # im.show()
        print(im.mode)
        print(im.size)
        print(im.format)

    @staticmethod
    def split(im, region):
        tmp_im = im.crop(region).resize((IM_W * 3, IM_H * 3))
        # tmp_im.show()
        im_set = []
        for i in range(3):
            for j in range(3):
                im_set.append(tmp_im.crop((IM_W * i, IM_H * j, IM_W * i + IM_W, IM_H * j + IM_H)))
        return im_set

    @staticmethod
    def get_coordinates(im):
        w, h = im.size
        last = (-10, -10)
        dots = []
        for i in range(w):
            for j in range(h):
                if im.getpixel((i, j)) != (0, 0, 0, 0):
                    if math.fabs(last[0] - i) > 7 or math.fabs(last[1] - j) > 7:
                        dots.append((i, j))
                        last = (i, j)
        # print(dots)
        remove_item = []
        for m in range(dots.__len__() - 1):
            for n in range(m + 1, dots.__len__()):
                if math.fabs(dots[m][0] - dots[n][0]) <= 7 and math.fabs(dots[m][1] - dots[n][1]) <= 7:
                    remove_item.append(dots[n])
        for k in remove_item:
            if k in dots:
                dots.remove(k)
        # print(dots)
        return dots

    @staticmethod
    def get_border(im):
        w, h = im.size
        min_x = w
        min_y = h
        max_x = -1
        max_y = -1
        for i in range(w):
            for j in range(h):
                if im.getpixel((i, j)) != (0, 0, 0, 0):
                    if i < min_x:
                        min_x = i
                    if i > max_x:
                        max_x = i
                    if j > max_y:
                        max_y = j
                    if j < min_y:
                        min_y = j
        return [min_x + 4, min_y + 4, max_x - 2, max_y - 2]

    @staticmethod
    def create_labelpad():
        var = multivariate_normal(mean=[0, 0], cov=[[COVSZ, 0], [0, COVSZ]])
        return np.array(
            [var.pdf([(x - HSZ), (y - HSZ)]) / COEF * COVSZ for x in range(SZ) for y in range(SZ)]).reshape((SZ, SZ))


if __name__ == '__main__':
    tr = TIMReader()
