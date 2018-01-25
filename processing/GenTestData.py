import numpy as np
import glob
import os
from PIL import Image
import openslide

TUMOR_TIF_SRC = "/home/suidong/Documents/camelyon16/TrainingData/Train_Tumor/"
TUMOR_MASK_TIF_SRC = "/home/suidong/Documents/camelyon16/TrainingData/Ground_Truth/Mask/"
TUMOR_PNG_ROOT = "/home/suidong/Documents/output/camelyon16/tumor/level/3/"
TUMOR_MASK_PNG_ROOT = "/home/suidong/Documents/output/camelyon16/mask/level/3/"


class GenTestData(object):
    def __init__(self, level):
        self.gen_tumor_png(level)
        self.gen_tumor_mask_png(level)

    def gen_tumor_png(self, level=6):
        tumor_img_file_list = glob.glob(os.path.join(TUMOR_TIF_SRC, '*.tif'))
        print(len(tumor_img_file_list))
        tumor_img_file_names = [os.path.splitext(os.path.split(img_path)[-1])[0] for img_path in tumor_img_file_list]
        for (index, tumor_img_file) in enumerate(tumor_img_file_list):
            slide = openslide.OpenSlide(tumor_img_file)
            [m_img, n_img] = slide.level_dimensions[level]
            tumor_tile = np.array(slide.read_region((m_img, n_img), level, (m_img, n_img)))
            if not os.path.exists(TUMOR_PNG_ROOT):
                os.mkdir(TUMOR_PNG_ROOT)
            im = Image.fromarray(tumor_tile.astype(np.uint8))
            im.save(os.path.join(TUMOR_PNG_ROOT, tumor_img_file_names[index] + ".png"), 'png')

    def gen_tumor_mask_png(self, level=6):
        tumor_mask_img_file_list = glob.glob(os.path.join(TUMOR_MASK_TIF_SRC, '*.tif'))
        print(len(tumor_mask_img_file_list))
        tumor_mask_img_file_names = [os.path.splitext(os.path.split(img_path)[-1])[0] for img_path in tumor_mask_img_file_list]
        for (index, tumor_mask_img_file) in enumerate(tumor_mask_img_file_list):
            mask = openslide.OpenSlide(tumor_mask_img_file)
            [m_img, n_img] = mask.level_dimensions[level]
            tumor_mask_tile = np.array(mask.read_region((m_img, n_img), level, (m_img, n_img)))
            if not os.path.exists(TUMOR_MASK_PNG_ROOT):
                os.mkdir(TUMOR_MASK_PNG_ROOT)
            im = Image.fromarray(tumor_mask_tile.astype(np.uint8))
            im.save(os.path.join(TUMOR_MASK_PNG_ROOT, tumor_mask_img_file_names[index] + ".png"), 'png')


if __name__ == '__main__':
    GenTestData(3)
