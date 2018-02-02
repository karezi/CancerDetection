from PIL import Image
import os
import glob
import random

LEVEL = 8
INPUT_ROOT = '/home/suidong/Documents/output/camelyon17/'
OUTPUT_ROOT = '/home/suidong/Documents/output/camelyon17/'
IMG_EDGE_DIR = INPUT_ROOT + 'patch_edge/' + str(LEVEL) + '/'
IMG_TUMOR_DIR = INPUT_ROOT + 'patch_tumor/' + str(LEVEL) + '/'
IMG_NORMAL_DIR = INPUT_ROOT + 'patch_normal/' + str(LEVEL) + '/'
MASK_EDGE_DIR = INPUT_ROOT + 'patch_edge_mask/' + str(LEVEL) + '/'
MASK_TUMOR_DIR = INPUT_ROOT + 'patch_tumor_mask/' + str(LEVEL) + '/'
MASK_NORMAL_DIR = INPUT_ROOT + 'patch_normal_mask/' + str(LEVEL) + '/'
SLIDE_TRAIN_DIR = OUTPUT_ROOT + 'slide_train/' + str(LEVEL) + '/'
SLIDE_TEST_DIR = OUTPUT_ROOT + 'slide_test/' + str(LEVEL) + '/'
MASK_TRAIN_DIR = OUTPUT_ROOT + 'mask_train/' + str(LEVEL) + '/'
MASK_TEST_DIR = OUTPUT_ROOT + 'mask_test/' + str(LEVEL) + '/'


class PatchRandomSelecet(object):
    def __init__(self):
        pass


    def select_img(self):
        edge_list = glob.glob(os.path.join(IMG_EDGE_DIR, '*.tif'))
        tumor_list = glob.glob(os.path.join(IMG_TUMOR_DIR, '*.tif'))
        normal_list = glob.glob(os.path.join(IMG_NORMAL_DIR, '*.tif'))
        count = len(edge_list) + len(tumor_list)
        normal_list = random.sample(normal_list, count)
        final_list = edge_list + tumor_list + normal_list
        random.shuffle(final_list)
        self.get_type(final_list)
        final_list_len = len(final_list)
        train_len = int(final_list_len * 0.8)
        train_list = final_list[0:train_len]
        test_list = final_list[train_len:final_list_len]
        # train_list save
        for index, file_path in enumerate(train_list):
            new_file_name = "train_" + str(index)
            im = Image.open(file_path)
            im.save(os.path.join(SLIDE_TRAIN_DIR, new_file_name))
            mask_path = self.find_mask_path(file_path)
            im2 = Image.open(mask_path)
            im2.save(os.path.join(MASK_TRAIN_DIR, new_file_name))
        # test_list save
        for index, file_path in enumerate(test_list):
            new_file_name = "test_" + str(index)
            im = Image.open(file_path)
            im.save(os.path.join(SLIDE_TEST_DIR, new_file_name))
            mask_path = self.find_mask_path(file_path)
            im2 = Image.open(mask_path)
            im2.save(os.path.join(MASK_TEST_DIR, new_file_name))


    def find_mask_path(self, url):
        bn = os.path.basename(url)
        dn = os.path.dirname(url)
        bn2 = os.path.basename(dn)
        dn2 = os.path.dirname(dn)
        return os.path.join(dn2 + '_mask', bn2, bn)


    def get_type(self, l):
        for index, file_name in enumerate(l):
            dn = os.path.dirname(file_name)
            dn2 = os.path.dirname(dn)
            type = os.path.basename(dn2)
            print(str(index) + ":" + type)


if __name__ == '__main__':
    PatchRandomSelecet().select_img()
