# Generate testing images with 128*128 in different levels.
import numpy
from PIL import Image
from pylab import *
import openslide
import os
import glob

IMG_SUFFIX = 'png'
SINGLE_WH = 128
LEVEL = 6

INPUT_ROOT = '/home/suidong/Documents/camelyon16/'
OUTPUT_ROOT = '/home/suidong/Documents/output/camelyon16/'

IMG_TIF_DIR = INPUT_ROOT + 'TrainingData/Train_Tumor/'
MASK_TIF_DIR = INPUT_ROOT + 'TrainingData/Ground_Truth/Mask/'

DST_IMG_EDGE_DIR = OUTPUT_ROOT + 'patch_edge/' + str(LEVEL) + '/'
DST_IMG_TUMOR_DIR = OUTPUT_ROOT + 'patch_tumor/' + str(LEVEL) + '/'
DST_IMG_NORMAL_DIR = OUTPUT_ROOT + 'patch_normal/' + str(LEVEL) + '/'
DST_MASK_EDGE_DIR = OUTPUT_ROOT + 'patch_edge_mask/' + str(LEVEL) + '/'
DST_MASK_TUMOR_DIR = OUTPUT_ROOT + 'patch_tumor_mask/' + str(LEVEL) + '/'
DST_MASK_NORMAL_DIR = OUTPUT_ROOT + 'patch_normal_mask/' + str(LEVEL) + '/'


class TifReader(object):
    _img_tif_files = []

    def __init__(self):
        self._img_tif_files = glob.glob(IMG_TIF_DIR + '*.tif')

    def read_tif_region(self):
        for img_file_url in self._img_tif_files:
            slide = openslide.OpenSlide(img_file_url)
            file_name = os.path.basename(img_file_url).split('.')[0]
            mask_url = os.path.join(MASK_TIF_DIR, file_name + '_Mask.tif')
            mask = openslide.OpenSlide(mask_url)
            print('Open:' + file_name)
            im_dim = slide.level_dimensions[LEVEL]
            mask_dim = mask.level_dimensions[LEVEL]
            print(im_dim)
            print(mask_dim)
            print('Check:' + file_name + ' successfully')
            split_x = range(0, mask_dim[0], SINGLE_WH)
            split_y = range(0, mask_dim[1], SINGLE_WH)
            count = 1
            total_num = (len(split_x) - 1) * (len(split_y) - 1)
            scale = slide.level_downsamples[LEVEL]
            for j in range(len(split_y) - 1):
                for i in range(len(split_x) - 1):
                    print('Handling:' + str(count) + '/' + str(total_num))
                    count += 1
                    fname = (file_name + '_' + str(LEVEL) + '_%09d.' + IMG_SUFFIX) % (count)
                    im_mask = mask.read_region((int(split_x[i] * scale), int(split_y[j] * scale)), LEVEL, (SINGLE_WH, SINGLE_WH))
                    res = self.judge(numpy.array(im_mask))
                    im_slide = slide.read_region((int(split_x[i] * scale), int(split_y[j] * scale)), LEVEL, (SINGLE_WH, SINGLE_WH))
                    if res == 3:
                        print('Store to edge folder')
                        if not os.path.exists(DST_IMG_EDGE_DIR):
                            os.makedirs(DST_IMG_EDGE_DIR)
                        if not os.path.exists(DST_MASK_EDGE_DIR):
                            os.makedirs(DST_MASK_EDGE_DIR)
                        im_slide.save(os.path.join(DST_IMG_EDGE_DIR, fname))
                        im_mask.save(os.path.join(DST_MASK_EDGE_DIR, fname))
                    elif res == 1:
                        print('Store to tumor folder')
                        if not os.path.exists(DST_IMG_TUMOR_DIR):
                            os.makedirs(DST_IMG_TUMOR_DIR)
                        if not os.path.exists(DST_MASK_TUMOR_DIR):
                            os.makedirs(DST_MASK_TUMOR_DIR)
                        im_slide.save(os.path.join(DST_IMG_TUMOR_DIR, fname))
                        im_mask.save(os.path.join(DST_MASK_TUMOR_DIR, fname))
                    elif res == 2:
                        print('Store to normal folder')
                        if not os.path.exists(DST_IMG_NORMAL_DIR):
                            os.makedirs(DST_IMG_NORMAL_DIR)
                        if not os.path.exists(DST_MASK_NORMAL_DIR):
                            os.makedirs(DST_MASK_NORMAL_DIR)
                        im_slide.save(os.path.join(DST_IMG_NORMAL_DIR, fname))
                        im_mask.save(os.path.join(DST_MASK_NORMAL_DIR, fname))
            slide.close()
            mask.close()

    # def read_mask_region(self):
    #     count = 1
    #     file_list = []
    #     for filename in os.listdir(DST_IMG_DIR):
    #         name = os.path.splitext(filename)[0];
    #         if len(name) == 6:
    #             file_list.append(int(name))
    #     for i in range(len(self.split_x) - 1):
    #         for j in range(len(self.split_y) - 1):
    #             if count in file_list:
    #                 mask_tile = numpy.array(self._mask.read_region((self.split_x[i], self.split_y[j]), 0, (self._single_wh, self._single_wh)))
    #                 name = ('%06d.' + IMG_SUFFIX) % (count)
    #                 self.matrix_to_image(mask_tile).save(os.path.join(DST_MASK_DIR, name))
    #             count += 1
        # plt.figure()
        # plt.imshow(tile)
        # plt.show()

    @staticmethod
    def matrix_to_image(data):
        new_im = Image.fromarray(data.astype(np.uint8))
        return new_im

    # @staticmethod
    # def judge(data):
    #     init_data = data[0][0][0] // 255
    #     all_or = init_data
    #     all_and = init_data
    #     for i in range(data.shape[0]):
    #         for j in range(data.shape[1]):
    #             if i == 0 and j == 0:
    #                 continue
    #             tmp_data = data[j][i][0] // 255
    #             if init_data != tmp_data:
    #                 return 3
    #             all_and &= tmp_data
    #             all_or |= tmp_data
    #     if all_and == 1:#11111
    #         return 1
    #     elif all_or == 0:#00000
    #         return 2
    #     else:
    #         return 3#10101

    @staticmethod
    def judge(data):
        init_data = data[0][0][0]
        if init_data == 0:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if data[j][i][0] == 255:
                        return 3 # 10101
            return 2 # 00000
        elif init_data == 255:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if data[j][i][0] == 0:
                        return 3 # 10101
            return 1 # 11111
        else:
            return -1


if __name__ == '__main__':
    TifReader().read_tif_region()
