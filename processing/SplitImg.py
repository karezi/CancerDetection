#coding:utf-8
import cv2
from matplotlib import pyplot as plt


class SplitImg(object):
    def __init__(self):
        im = cv2.imread('/Users/karezi/Desktop/1.png')
        self.otsu(im)

    def otsu(self, im):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        plt.subplot(121), plt.imshow(im, "gray")
        plt.title("source image"), plt.xticks([]), plt.yticks([])
        # plt.subplot(132), plt.hist(im.ravel(), 256)
        # plt.title("Histogram"), plt.xticks([]), plt.yticks([])
        ret1, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # 方法选择为THRESH_OTSU
        plt.subplot(122), plt.imshow(th1, "gray")
        plt.title("OTSU, threshold is " + str(ret1)), plt.xticks([]), plt.yticks([])
        plt.show()


if __name__ == '__main__':
    pi = SplitImg()
