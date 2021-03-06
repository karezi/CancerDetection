import numpy
from PIL import Image
from pylab import *
import openslide
import os
import glob

IMG_SUFFIX = 'png'
SINGLE_WH = 244

INPUT_ROOT = '/Volumes/KAREZI/'
OUTPUT_ROOT = '/User/karezi/Desktop/'

IMG_TIF_DIR = INPUT_ROOT + 'camelyon16/TrainingData/Train_Tumor/'
MASK_TIF_DIR = INPUT_ROOT + 'camelyon16/TrainingData/Ground_Truth/Mask/'

DST_IMG_EDGE_DIR = OUTPUT_ROOT + 'output/edge/'
DST_IMG_TUMOR_DIR = OUTPUT_ROOT + 'output/tumor/'
DST_MASK_EDGE_DIR = OUTPUT_ROOT + 'output/edge_mask/'
DST_MASK_TUMOR_DIR = OUTPUT_ROOT + 'output/tumor_mask/'


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
            im_dim = slide.dimensions
            mask_dim = mask.dimensions
            if im_dim == mask_dim:
                print('Check:' + file_name + ' successfully')
                split_x = range(0, im_dim[0], SINGLE_WH)
                split_y = range(0, im_dim[1], SINGLE_WH)
                count = 1
                total_num = (len(split_x) - 1) * (len(split_y) - 1)
                for i in range(len(split_x) - 1):
                    for j in range(len(split_y) - 1):
                        print('Handling:' + str(count) + '/' + str(total_num))
                        count += 1
                        fname = ('%09d.' + IMG_SUFFIX) % (count)
                        mask_tile = numpy.array(mask.read_region((split_x[i], split_y[j]), 0, (SINGLE_WH, SINGLE_WH)))
                        res = self.judge(mask_tile)
                        if res == 3:
                            print('Store to edge folder')
                            slide_tile = numpy.array(slide.read_region((split_x[i], split_y[j]), 0, (SINGLE_WH, SINGLE_WH)))
                            im_slide = self.matrix_to_image(slide_tile)
                            im_mask = self.matrix_to_image(mask_tile)
                            if not os.path.exists(DST_IMG_EDGE_DIR):
                                os.makedirs(DST_IMG_EDGE_DIR)
                            new_path_img = os.path.join(DST_IMG_EDGE_DIR, file_name)
                            if not os.path.exists(DST_MASK_EDGE_DIR):
                                os.makedirs(DST_MASK_EDGE_DIR)
                            new_path_mask = os.path.join(DST_MASK_EDGE_DIR, file_name)
                            if not os.path.exists(new_path_img):
                                os.makedirs(new_path_img)
                            if not os.path.exists(new_path_mask):
                                os.makedirs(new_path_mask)
                            im_slide.save(os.path.join(new_path_img, fname))
                            im_mask.save(os.path.join(new_path_mask, fname))
                        elif res == 1:
                            print('Store to tumor folder')
                            slide_tile = numpy.array(slide.read_region((split_x[i], split_y[j]), 0, (SINGLE_WH, SINGLE_WH)))
                            im_slide = self.matrix_to_image(slide_tile)
                            im_mask = self.matrix_to_image(mask_tile)
                            new_path_img = os.path.join(DST_IMG_TUMOR_DIR, file_name)
                            new_path_mask = os.path.join(DST_MASK_TUMOR_DIR, file_name)
                            if not os.path.exists(new_path_img):
                                os.makedirs(new_path_img)
                            if not os.path.exists(new_path_mask):
                                os.makedirs(new_path_mask)
                            im_slide.save(os.path.join(new_path_img, fname))
                            im_mask.save(os.path.join(new_path_mask, fname))
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

