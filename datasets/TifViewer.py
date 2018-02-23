import multiresolutionimageinterface as mir
import numpy as np
from PIL import Image
from pylab import *


class TifViewer(object):
    def __init__(self):
        reader = mir.MultiResolutionImageReader()
        self.slide = reader.open("/home/suidong/Documents/camelyon17_data_backup/slide/patient_010_node_4.tif")
        self.mask = reader.open("/home/suidong/Documents/camelyon17_data_backup/mask/patient_010_node_4.tif")

    def __del__(self):
        self.slide.close()
        self.mask.close()

    def show_whole(self, level):
        [m_img, n_img] = self.slide.getLevelDimensions(level)
        [m_mask, n_mask] = self.mask.getLevelDimensions(level)
        print (m_img, n_img)
        print (m_mask, n_mask)
        tile_img = np.array(self.slide.getUCharPatch(0, 0, int(m_img), int(n_img), level))
        tile_mask = np.array(self.mask.getUCharPatch(0, 0, int(m_mask), int(n_mask), level))
        plt.figure(1)
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        plt.sca(ax1)
        plt.imshow(tile_img)
        plt.sca(ax2)
        plt.imshow(np.squeeze(tile_mask, axis=(2,)))
        plt.show()

    def show_region(self, region, level):
        [m_img, n_img] = self.slide.getLevelDimensions(level)
        [m_mask, n_mask] = self.mask.getLevelDimensions(level)
        print (m_img, n_img)
        print (m_mask, n_mask)
        scale = self.slide.getLevelDownsample(level)
        tile_img = np.array(self.slide.getUCharPatch(int(region[0] * scale), int(region[1] * scale), int(region[2]), int(region[3]), level))
        tile_mask = np.array(self.mask.getUCharPatch(int(region[0] * scale), int(region[1] * scale), int(region[2]), int(region[3]), level))
        plt.figure(1)
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        plt.sca(ax1)
        plt.imshow(tile_img)
        plt.sca(ax2)
        plt.imshow(np.squeeze(tile_mask, axis=(2,)))
        plt.show()


if __name__ == '__main__':
    tv = TifViewer()
    tv.show_region((1000, 8000, 500, 500), 6)
