import numpy
from PIL import Image
from pylab import *
import openslide


class TifSearch(object):

    def __init__(self):
        self.tif_mask_files = "/Users/karezi/Desktop/Test_001_Mask.tif"
        self.tif_img_files = "/Users/karezi/Desktop/Test_001.tif"

    def show_region(self, level = 0):
        slide = openslide.OpenSlide(self.tif_img_files)
        mask = openslide.OpenSlide(self.tif_mask_files)
        level_count_slide = slide.level_count
        level_count_mask = mask.level_count
        print(level_count_slide)
        print(level_count_mask)
        [m_img, n_img] = slide.level_dimensions[level]
        [m_mask, n_mask] = mask.level_dimensions[level]
        print (m_img, n_img)
        print (m_mask, n_mask)
        tile_img = numpy.array(slide.read_region((m_img, n_img), level, (m_img, n_img)))
        tile_mask = numpy.array(mask.read_region((m_mask, n_mask), level, (m_mask, n_mask)))
        plt.figure(1)
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        plt.sca(ax1)
        plt.imshow(tile_img)
        plt.sca(ax2)
        plt.imshow(tile_mask)
        plt.show()
        slide.close()
        mask.close()

    @staticmethod
    def matrix_to_image(data):
        new_im = Image.fromarray(data.astype(np.uint8))
        return new_im

if __name__ == '__main__':
    ts = TifSearch()
    ts.show_region(6)
