# Multi-level detection
import multiresolutionimageinterface as mir
import os
import numpy as np
from PIL import Image
from networks.ColorFCN import *
from pylab import *

IMG_SUFFIX = 'png'
SINGLE_WH = 128
TOP_LEVEL = 6
NAME = "patient_004_node_4"
INPUT_ROOT = '/home/suidong/Documents/camelyon17_data_backup/'
OUTPUT_ROOT = '/home/suidong/Documents/output/camelyon17/tiftest/' + NAME
ORIGIN_TIF_URL = INPUT_ROOT + "slide/" + NAME + ".tif"
MASK_TIF_URL = INPUT_ROOT + "mask/" + NAME + ".tif"
LOG_ROOT = '/home/suidong/Documents/output/camelyon17/log/'
H5_NAME = ['0', '1', '2.1518497504', '3.1518416911', '4.1518181780', '5.1518172234', '6.1519058660']


class MultiLevelDetection(object):
    _reader = None
    _origin = None
    _mask = None
    _check = True
    _dims = []
    _detection_result = [[] for i in range(TOP_LEVEL + 1)]
    _m = 0
    _n = 0
    _models = [None, None]

    def __init__(self):
        self.prepare_model()
        self._reader = mir.MultiResolutionImageReader()
        self.input()
        if self._check:
            target_patches = self.split_whole_level(TOP_LEVEL)
            self.detection(target_patches, TOP_LEVEL - 1)
        print(self._detection_result)

    def __del__(self):
        self._origin.close()
        self._mask.close()

    def prepare_model(self):
        for i in range(2, TOP_LEVEL + 1):
            self._models.append(self.load_weight(i))

    def input(self):
        self._origin = self._reader.open(ORIGIN_TIF_URL)
        self._mask = self._reader.open(MASK_TIF_URL)
        for i in range(TOP_LEVEL + 1):
            im_dim = self._origin.getLevelDimensions(i)
            mask_dim = self._mask.getLevelDimensions(i)
            print(im_dim)
            print(mask_dim)
            print(im_dim == mask_dim)
            self._dims.append(im_dim)
            if im_dim != mask_dim:
                self._check = False

    def split_whole_level(self, level):
        output_url = os.path.join(OUTPUT_ROOT, str(level))
        split_x = range(0, self._dims[level][0], SINGLE_WH)
        split_y = range(0, self._dims[level][1], SINGLE_WH)
        count = 0
        self._m = len(split_x) - 1
        self._n = len(split_y) - 1
        total_num = self._m * self._n
        scale = self._origin.getLevelDownsample(level)
        for j in range(len(split_y) - 1):
            for i in range(len(split_x) - 1):
                print('Handling:' + str(count) + '/' + str(total_num - 1))
                count += 1
                slide_file_name = ('%d.slide.' + IMG_SUFFIX) % count
                mask_file_name = ('%d.mask.' + IMG_SUFFIX) % count
                dm_file_name = ('%d.dm.' + IMG_SUFFIX) % count
                im_slide = self._origin.getUCharPatch(int(split_x[i] * scale), int(split_y[j] * scale), SINGLE_WH,
                                                      SINGLE_WH, level)
                res = self.detect_one_patch(level, im_slide, os.path.join(output_url, dm_file_name))
                if res == 1 or res == 0.5:
                    self._detection_result[level].append("%s:%s" % (str(count), str(res)))
                # Visualization
                im_mask = self._mask.getUCharPatch(int(split_x[i] * scale), int(split_y[j] * scale), SINGLE_WH,
                                                   SINGLE_WH, level)
                self.matrix_to_image(im_slide).save(os.path.join(output_url, slide_file_name))
                self.matrix_to_image(np.squeeze(im_mask, axis=(2,))).save(os.path.join(output_url, mask_file_name))
        return self._detection_result[level]

    def split_one_patch(self, level, last_no):
        output_url = os.path.join(OUTPUT_ROOT, str(level))
        last_s = 2 ** (TOP_LEVEL - level - 1)
        last_m = self._m * last_s
        a = last_no / last_m
        b = last_no % last_m
        x = 2 * a * SINGLE_WH
        y = 2 * b * SINGLE_WH
        scale = self._origin.getLevelDownsample(level)
        no1 = a * last_m * 4 + b * 2 + 1
        no2 = a * last_m * 4 + b * 2 + 2
        no3 = (a * 2 + 1) * last_m * 2 + b * 2 + 1
        no4 = (a * 2 + 1) * last_m * 2 + b * 2 + 2
        im_slide_1 = self._origin.getUCharPatch(
            int(x * scale), int(y * scale), SINGLE_WH, SINGLE_WH, level)
        im_slide_2 = self._origin.getUCharPatch(
            int((x + SINGLE_WH) * scale), int(y * scale), SINGLE_WH, SINGLE_WH, level)
        im_slide_3 = self._origin.getUCharPatch(
            int(x * scale), int((y + SINGLE_WH) * scale), SINGLE_WH, SINGLE_WH, level)
        im_slide_4 = self._origin.getUCharPatch(
            int((x + SINGLE_WH) * scale), int((y + SINGLE_WH) * scale), SINGLE_WH, SINGLE_WH, level)
        res = []
        res1 = self.detect_one_patch(level, im_slide_1, os.path.join(output_url, ('%d.%d.dm.' + IMG_SUFFIX) % (no1, last_no)))
        if res1 == 1 or res1 == 0.5:
            res.append("%s:%s" % (str(no1), str(res1)))
            self._detection_result[level].append("%s:%s" % (str(no1), str(res1)))
        res2 = self.detect_one_patch(level, im_slide_2, os.path.join(output_url, ('%d.%d.dm.' + IMG_SUFFIX) % (no2, last_no)))
        if res2 == 1 or res2 == 0.5:
            res.append("%s:%s" % (str(no2), str(res2)))
            self._detection_result[level].append("%s:%s" % (str(no2), str(res2)))
        res3 = self.detect_one_patch(level, im_slide_3, os.path.join(output_url, ('%d.%d.dm.' + IMG_SUFFIX) % (no3, last_no)))
        if res3 == 1 or res3 == 0.5:
            res.append("%s:%s" % (str(no3), str(res3)))
            self._detection_result[level].append("%s:%s" % (str(no3), str(res3)))
        res4 = self.detect_one_patch(level, im_slide_4, os.path.join(output_url, ('%d.%d.dm.' + IMG_SUFFIX) % (no4, last_no)))
        if res4 == 1 or res4 == 0.5:
            res.append("%s:%s" % (str(no4), str(res4)))
            self._detection_result[level].append("%s:%s" % (str(no4), str(res4)))
        # Visualization
        im_mask_1 = self._mask.getUCharPatch(
            int(x * scale), int(y * scale), SINGLE_WH, SINGLE_WH, level)
        im_mask_2 = self._mask.getUCharPatch(
            int((x + SINGLE_WH) * scale), int(y * scale), SINGLE_WH, SINGLE_WH, level)
        im_mask_3 = self._mask.getUCharPatch(
            int(x * scale), int((y + SINGLE_WH) * scale), SINGLE_WH, SINGLE_WH, level)
        im_mask_4 = self._mask.getUCharPatch(
            int((x + SINGLE_WH) * scale), int((y + SINGLE_WH) * scale), SINGLE_WH, SINGLE_WH, level)
        self.matrix_to_image(im_slide_1).save(
            os.path.join(output_url, ('%d.%d.slide.' + IMG_SUFFIX) % (no1, last_no)))
        self.matrix_to_image(np.squeeze(im_mask_1, axis=(2,))).save(
            os.path.join(output_url, ('%d.%d.mask.' + IMG_SUFFIX) % (no1, last_no)))
        self.matrix_to_image(im_slide_2).save(
            os.path.join(output_url, ('%d.%d.slide.' + IMG_SUFFIX) % (no2, last_no)))
        self.matrix_to_image(np.squeeze(im_mask_2, axis=(2,))).save(
            os.path.join(output_url, ('%d.%d.mask.' + IMG_SUFFIX) % (no2, last_no)))
        self.matrix_to_image(im_slide_3).save(
            os.path.join(output_url, ('%d.%d.slide.' + IMG_SUFFIX) % (no3, last_no)))
        self.matrix_to_image(np.squeeze(im_mask_3, axis=(2,))).save(
            os.path.join(output_url, ('%d.%d.mask.' + IMG_SUFFIX) % (no3, last_no)))
        self.matrix_to_image(im_slide_4).save(
            os.path.join(output_url, ('%d.%d.slide.' + IMG_SUFFIX) % (no4, last_no)))
        self.matrix_to_image(np.squeeze(im_mask_4, axis=(2,))).save(
            os.path.join(output_url, ('%d.%d.mask.' + IMG_SUFFIX) % (no4, last_no)))
        return res

    @staticmethod
    def matrix_to_image(data, mode=None):
        new_im = Image.fromarray(data.astype(np.uint8), mode)
        return new_im

    def detection(self, target_patches, level):
        if level <= 1 or target_patches == []:
            return level
        else:
            for patch in target_patches:
                no = int(patch.split(':')[0])
                print(str(no))
                res_target_patches = self.split_one_patch(level, no)
                if level == 2:
                    continue
                else:
                    self.detection(res_target_patches, level - 1)

    @staticmethod
    def load_weight(level):
        model = ColorFCN().get_model()
        weight_path = os.path.join(LOG_ROOT, "MFCNInterface." + H5_NAME[level] + ".h5")
        model.load_weights(weight_path)
        return model

    def detect_one_patch(self, level, test_patch, output_path):
        predict_target = np.array([array(test_patch)]) / 255.0
        res = self._models[level].predict(predict_target)
        if output_path:
            res_image = res[0][..., 0] * 255.0
            self.matrix_to_image(res_image).save(output_path)
        print(np.sum(res))
        print(np.max(res))
        if np.sum(res) > 1000:
            return 1  # patch_edge(edge-big)
        else:
            if np.max(res) > 0.17:
                return 0.5  # patch_edge(edge-small)
            else:
                return 0  # patch_normal(normal)


if __name__ == "__main__":
    MultiLevelDetection()
