import numpy as np
import glob
import os
from networks.ColorFCN import *
import time
from pylab import *
from PIL import Image
from postprocessing.LossHistory import *
from keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 32
EPOCHS = 5
MODE = 2  # 0:train without augment, 1:train with augment, 2:test
LOG_ROOT = '/home/suidong/Documents/output/camelyon17/log/'
INPUT_ROOT = '/home/suidong/Documents/output/tim2015/'
PATCH_DIR = INPUT_ROOT + "learning/images/"
MAP_DIR = INPUT_ROOT + "learning/maps/"
H5_PATH = os.path.join(LOG_ROOT, "TIMCancerDetection." + str(int(time.time())) + ".h5")
H5_PATH_GEN = os.path.join(LOG_ROOT, "TIMCancerDetection.1519400153.h5")
NO = "001.1"


class TIMCancerDetection(object):
    x_train = None
    y_train = None
    x_test = None
    y_test = None

    def __init__(self):
        model = ColorFCN().get_model()
        if MODE == 1:
            self.prepare_data()
            self.train_with_augment(model)
        elif MODE == 2:
            model.load_weights(H5_PATH_GEN)
            self.test(model, NO)

    def prepare_data(self):
        file_list = glob.glob(os.path.join(PATCH_DIR, '*.png'))
        mask_list = glob.glob(os.path.join(MAP_DIR, '*.png'))
        images = np.array([array(Image.open(x).convert('RGB')) for x in file_list])
        masks = np.array([array(Image.open(x).convert('L'))[..., newaxis] for x in mask_list])
        images = images / 255.0
        masks = masks / 255.0
        print(images.shape)
        print(masks.shape)
        total_len = images.shape[0]
        train_len = int(total_len * 0.8)
        self.x_train, self.y_train = images[0:train_len], masks[0:train_len]
        self.x_test, self.y_test = images[train_len:total_len], masks[train_len:total_len]

    def train_with_augment(self, model):
        history = LossHistory()
        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)
        datagen.fit(self.x_train)
        model.fit_generator(datagen.flow(self.x_train, self.y_train, batch_size=BATCH_SIZE),
                            steps_per_epoch=len(self.x_train), epochs=EPOCHS, callbacks=[history])
        model.save(H5_PATH)
        score = model.evaluate(self.x_test, self.y_test, batch_size=BATCH_SIZE, verbose=1, sample_weight=None)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        history.loss_plot('epoch')

    @staticmethod
    def test(model, no):
        test_image = Image.open(os.path.join(PATCH_DIR, no + '.png')).convert('RGB')
        test_mask = Image.open(os.path.join(MAP_DIR, no + '.png')).convert('L')
        predict_target = np.array([array(test_image)]) / 255.0
        res = model.predict(predict_target)
        res_image = res[0][..., 0] * 255.0
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.title('input')
        plt.imshow(test_image)
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.title('output')
        plt.imshow(res_image)
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.title('mask')
        plt.imshow(test_mask)
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    TIMCancerDetection()
