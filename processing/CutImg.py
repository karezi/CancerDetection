import os
import glob
from PIL import Image

IMG_SQUARE_PATH = "/Users/karezi/Desktop/datasets/tumor/camelyon16trainingdata/tumor_square/"
MAP_SQUARE_PATH = "/Users/karezi/Desktop/datasets/tumor/camelyon16trainingdata/mask_square/"
TUMOR_PNG_ROOT = "/Users/karezi/Desktop/datasets/tumor/camelyon16trainingdata/tumor"
TUMOR_MASK_PNG_ROOT = "/Users/karezi/Desktop/datasets/tumor/camelyon16trainingdata/mask"


class CutImg(object):
    def __init__(self):
        self.cut()

    def cut(self):
        tumor_png_file_list = glob.glob(os.path.join(TUMOR_PNG_ROOT, '*.png'))
        mask_png_file_list = glob.glob(os.path.join(TUMOR_MASK_PNG_ROOT, '*.png'))
        print(len(tumor_png_file_list))
        for f in tumor_png_file_list:
            im = Image.open(f)
            w, h = im.size
            print(im.size)
            if w > h:
                box1 = (0, 0, w/2, h)
                box2 = (w/2, 0, w, h)
            else:
                box1 = (0, 0, w, h/2)
                box2 = (0, h/2, w, h)
            print("box1:" + str(box1))
            print("box2:" + str(box2))
            region1 = self.square(im.crop(box1))
            region2 = self.square(im.crop(box2))
            print("region1:" + str(region1.size))
            print("region2:" + str(region2.size))
            name = os.path.splitext(os.path.split(f)[1])[0]
            region1.save(os.path.join(IMG_SQUARE_PATH, name + '_1.png'))
            region2.save(os.path.join(IMG_SQUARE_PATH, name + '_2.png'))
        for mf in mask_png_file_list:
            im = Image.open(mf)
            w, h = im.size
            if w > h:
                box1 = (0, 0, w/2, h)
                box2 = (w/2, 0, w, h)
            else:
                box1 = (0, 0, w, h/2)
                box2 = (0, h/2, w, h)
            region1 = self.square(im.crop(box1))
            region2 = self.square(im.crop(box2))
            name = os.path.splitext(os.path.split(mf)[1])[0]
            region1.save(os.path.join(MAP_SQUARE_PATH, name + '_1.png'))
            region2.save(os.path.join(MAP_SQUARE_PATH, name + '_2.png'))

    def square(self, im):
        w, h = im.size
        print(im.size)
        if w > h:
            return im.crop(((w-h)/2, 0, h+(w-h)/2, h))
        else:
            return im.crop((0, (h-w)/2, w, w+(h-w)/2))


if __name__ == '__main__':
    CutImg()

