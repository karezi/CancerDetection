# Generate testing images with 128*128 in different levels from Camelyon 2017.
import numpy
from PIL import Image
from pylab import *
import multiresolutionimageinterface as mir
import os
import glob

IMG_SUFFIX = 'png'
SINGLE_WH = 128
LEVEL = 5

INPUT_ROOT = '/home/suidong/Documents/camelyon17_data_backup/'
OUTPUT_ROOT = '/home/suidong/Documents/output/camelyon17/'

IMG_TIF_DIR = INPUT_ROOT + 'slide/'
MASK_TIF_DIR = INPUT_ROOT + 'mask/'

DST_IMG_EDGE_DIR = OUTPUT_ROOT + 'patch_edge/' + str(LEVEL) + '/'
DST_IMG_TUMOR_DIR = OUTPUT_ROOT + 'patch_tumor/' + str(LEVEL) + '/'
DST_IMG_NORMAL_DIR = OUTPUT_ROOT + 'patch_normal/' + str(LEVEL) + '/'
DST_MASK_EDGE_DIR = OUTPUT_ROOT + 'patch_edge_mask/' + str(LEVEL) + '/'
DST_MASK_TUMOR_DIR = OUTPUT_ROOT + 'patch_tumor_mask/' + str(LEVEL) + '/'
DST_MASK_NORMAL_DIR = OUTPUT_ROOT + 'patch_normal_mask/' + str(LEVEL) + '/'


class Tif2Patch(object):
    _img_tif_files = []
    _reader = None

    def __init__(self):
        self._mask_tif_files = glob.glob(MASK_TIF_DIR + '*.tif')
        self._reader = mir.MultiResolutionImageReader()

    def gen_patches(self):
        for mask_file_url in self._mask_tif_files:
            mask = self._reader.open(mask_file_url)
            file_name = os.path.basename(mask_file_url).split('.')[0]
            slide_url = os.path.join(IMG_TIF_DIR, file_name + '.tif')
            slide = self._reader.open(slide_url)
            print('Open:' + file_name)
            if slide is None or mask is None:
                continue
            im_dim = slide.getLevelDimensions(LEVEL)
            mask_dim = mask.getLevelDimensions(LEVEL)
            print(im_dim)
            print(mask_dim)
            if im_dim != mask_dim or not im_dim or not mask_dim:
                continue
            print('Check:' + file_name + ' successfully')
            split_x = range(0, mask_dim[0], SINGLE_WH)
            split_y = range(0, mask_dim[1], SINGLE_WH)
            count = 1
            total_num = (len(split_x) - 1) * (len(split_y) - 1)
            scale = slide.getLevelDownsample(LEVEL)
            for j in range(len(split_y) - 1):
                for i in range(len(split_x) - 1):
                    print('Handling:' + str(count) + '/' + str(total_num))
                    count += 1
                    fname = (file_name + '_' + str(LEVEL) + '_%09d.' + IMG_SUFFIX) % (count)
                    im_mask = mask.getUCharPatch(int(split_x[i] * scale), int(split_y[j] * scale), SINGLE_WH, SINGLE_WH, LEVEL)
                    res = self.judge(numpy.array(im_mask))
                    im_slide = slide.getUCharPatch(int(split_x[i] * scale), int(split_y[j] * scale), SINGLE_WH, SINGLE_WH, LEVEL)
                    im_slide = im_slide
                    print im_mask.shape
                    print im_slide.shape
                    if res == 3:
                        print('Store to edge folder')
                        if not os.path.exists(DST_IMG_EDGE_DIR):
                            os.makedirs(DST_IMG_EDGE_DIR)
                        if not os.path.exists(DST_MASK_EDGE_DIR):
                            os.makedirs(DST_MASK_EDGE_DIR)
                        self.matrix_to_image(im_slide).save(os.path.join(DST_IMG_EDGE_DIR, fname))
                        self.matrix_to_image(np.squeeze(im_mask, axis=(2,))).save(os.path.join(DST_MASK_EDGE_DIR, fname))
                    elif res == 1:
                        print('Store to tumor folder')
                        if not os.path.exists(DST_IMG_TUMOR_DIR):
                            os.makedirs(DST_IMG_TUMOR_DIR)
                        if not os.path.exists(DST_MASK_TUMOR_DIR):
                            os.makedirs(DST_MASK_TUMOR_DIR)
                        self.matrix_to_image(im_slide).save(os.path.join(DST_IMG_TUMOR_DIR, fname))
                        self.matrix_to_image(np.squeeze(im_mask, axis=(2,))).save(os.path.join(DST_MASK_TUMOR_DIR, fname))
                    elif res == 2:
                        print('Store to normal folder')
                        if not os.path.exists(DST_IMG_NORMAL_DIR):
                            os.makedirs(DST_IMG_NORMAL_DIR)
                        if not os.path.exists(DST_MASK_NORMAL_DIR):
                            os.makedirs(DST_MASK_NORMAL_DIR)
                        self.matrix_to_image(im_slide).save(os.path.join(DST_IMG_NORMAL_DIR, fname))
                        self.matrix_to_image(np.squeeze(im_mask, axis=(2,))).save(os.path.join(DST_MASK_NORMAL_DIR, fname))
            slide.close()
            mask.close()

    @staticmethod
    def matrix_to_image(data, mode=None):
        # print(data.astype(np.uint8))
        new_im = Image.fromarray(data.astype(np.uint8), mode)
        return new_im

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
    Tif2Patch().gen_patches()
