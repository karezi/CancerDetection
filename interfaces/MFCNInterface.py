import os
from PIL import Image
import glob
from pylab import *
import numpy as np
from networks.ColorFCN import *
from networks.SimpleFCN import *
import time
from keras.callbacks import EarlyStopping
from postprocessing.LossHistory import *
from keras.preprocessing.image import ImageDataGenerator

LEVEL = 2
BATCH_SIZE = 32
EPOCHS = 5
TYPE = 1  # Is color
MODE = 3  # 0:train without augment, 1:train with augment, 2:test, 3:test_all
INPUT_ROOT = '/home/suidong/Documents/output/camelyon17/'
LOG_ROOT = '/home/suidong/Documents/output/camelyon17/log/'
SLIDE_TRAIN_DIR = INPUT_ROOT + 'slide_train/' + str(LEVEL) + '/'
SLIDE_TEST_DIR = INPUT_ROOT + 'slide_test/' + str(LEVEL) + '/'
MASK_TRAIN_DIR = INPUT_ROOT + 'mask_train/' + str(LEVEL) + '/'
MASK_TEST_DIR = INPUT_ROOT + 'mask_test/' + str(LEVEL) + '/'
H5_PATH = os.path.join(LOG_ROOT, "MFCNInterface." + str(LEVEL) + "." + str(int(time.time())) + ".h5")
H5_PATH_GEN_5 = os.path.join(LOG_ROOT, "MFCNInterface.5.1518172234.good.h5")
H5_PATH_GEN_4 = os.path.join(LOG_ROOT, "MFCNInterface.4.1518181780.h5")
H5_PATH_GEN_3 = os.path.join(LOG_ROOT, "MFCNInterface.3.1518416911.h5")
H5_PATH_GEN = os.path.join(LOG_ROOT, "MFCNInterface.2.1518497504.h5")
TEST_IMG_NO = 9
TEST_IMG_URL = SLIDE_TEST_DIR + '/test_' + str(TEST_IMG_NO)
TEST_MASK_URL = MASK_TEST_DIR + '/test_' + str(TEST_IMG_NO)
TEST_IMG_URL_ALL = SLIDE_TEST_DIR + '/test_'
TEST_MASK_URL_ALL = MASK_TEST_DIR + '/test_'


class MFCNInterface(object):
    x_train = None
    y_train = None
    x_test = None
    y_test = None
    category = []

    def __init__(self):
        self.get_class()
        self.prepare_data()
        print("Load Model")
        if TYPE:
            model = ColorFCN().get_model()
        else:
            model = SimpleFCN().get_model()
        if MODE == 0:
            self.prepare_data()
            self.train_without_augment(model)
        elif MODE == 1:
            self.prepare_data()
            self.train_with_augment(model)
        elif MODE == 2:
            model.load_weights(H5_PATH_GEN)
            self.test(model)
        elif MODE == 3:
            model.load_weights(H5_PATH_GEN)
            self.test_all(model)

    def prepare_data(self):
        file_list = glob.glob(os.path.join(SLIDE_TRAIN_DIR, '*'))
        mask_list = glob.glob(os.path.join(MASK_TRAIN_DIR, '*'))
        if TYPE:
            images = np.array([array(Image.open(x)) for x in file_list])
        else:
            images = np.array([array(Image.open(x).convert("L"))[..., newaxis] for x in file_list])
        masks = np.array([array(Image.open(x))[..., newaxis] for x in mask_list])
        images = images / 255.0
        masks = masks / 255.0
        print(images.shape)
        print(masks.shape)
        total_len = images.shape[0]
        train_len = int(total_len * 0.8)
        self.x_train, self.y_train = images[0:train_len], masks[0:train_len]
        self.x_test, self.y_test = images[train_len:total_len], masks[train_len:total_len]

    def get_class(self):
        with open(os.path.join(LOG_ROOT, 'train_test_' + str(LEVEL))) as file_object:
            for line in file_object:
                self.category.append(line.rstrip('\n').split(':')[1])
        print(len(self.category))

    def train_without_augment(self, model):
        history = LossHistory()
        # early_stopping = EarlyStopping(monitor='val_loss', patience=1)
        # model.fit(train_x, train_y, batch_size=4, epochs=1, validation_split=0.2, callbacks=[early_stopping])
        model.fit(self.x_train, self.y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(self.x_test, self.y_test), callbacks=[history])
        model.save(H5_PATH)
        score = model.evaluate(self.x_test, self.y_test, batch_size=BATCH_SIZE, verbose=1, sample_weight=None)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        history.loss_plot('epoch')

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
        # here's a more "manual" example
        # for e in range(EPOCHS):
        #     print 'Epoch', e
        #     batches = 0
        #     for x_batch, y_batch in datagen.flow(train_x, train_y, batch_size=32):
        #         loss = model.train(x_batch, y_batch)
        #         batches += 1
        #         if batches >= len(train_x) * 32:
        #             # we need to break the loop by hand because
        #             # the generator loops indefinitely
        #             break

    def test(self, model):
        print("type:" + self.category[len(self.x_train) + len(self.x_test) + TEST_IMG_NO])
        if TYPE:
            test_image = Image.open(TEST_IMG_URL)
            test_mask = Image.open(TEST_MASK_URL)
            predict_target = np.array([array(test_image)]) / 255.0
        else:
            test_image = Image.open(TEST_IMG_URL).convert("L")
            test_mask = Image.open(TEST_MASK_URL).convert("L")
            predict_target = np.array([array(test_image)[..., newaxis]]) / 255.0
        res = model.predict(predict_target)
        print(np.sum(res))
        print(np.max(res))
        res_image = res[0][..., 0] * 255.0
        if TYPE:
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
        else:
            plt.figure()
            plt.subplot(1, 3, 1)
            plt.title('input')
            plt.imshow(test_image, cmap='gray')
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.title('output')
            plt.imshow(res_image, cmap='gray')
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.title('mask')
            plt.imshow(test_mask, cmap='gray')
            plt.axis('off')
            plt.show()

    def test_all(self, model):
        test_list = glob.glob(os.path.join(SLIDE_TEST_DIR, '*'))
        count_all = len(test_list)
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        for i in range(len(test_list)):
            print(str(i))
            test_img_url = os.path.join(SLIDE_TEST_DIR, 'test_' + str(i))
            test_mask_url = os.path.join(MASK_TEST_DIR, 'test_' + str(i))
            if TYPE:
                test_image = Image.open(test_img_url)
                predict_target = np.array([array(test_image)]) / 255.0
            else:
                pass  # todosth
            res = model.predict(predict_target)
            print(np.sum(res))
            print(np.max(res))
            type_name = self.category[len(self.x_train) + len(self.x_test) + i]
            t = -1
            if type_name == "patch_edge":
                t = 0
            elif type_name == "patch_normal":
                t = 1
            elif type_name == "patch_tumor":
                t = 2
            print("type:" + type_name + ";" + str(t))
            t2 = -1
            if np.sum(res) > 1000:
                print("type:patch_edge(edge-big)")
                t2 = 2
            else:
                if np.max(res) > 0.17:
                    print("type:patch_edge(edge-small)")
                    t2 = 0
                else:
                    print("type:patch_normal(normal)")
                    t2 = 1
            if t == t2:
                print("nice")
                count1 = count1 + 1
            else:
                if (t == 0 or t == 2) and t2 == 1:
                    print("bad")
                    count2 = count2 + 1
                elif t == 1 and (t2 == 0 or t2 == 2):
                    print("just so so")
                    count3 = count3 + 1
                else:
                    print("good")
                    count4 = count4 + 1
            print("\n")
        print("nice:" + str(count1))
        print("bad:" + str(count2))
        print("just so so:" + str(count3))
        print("good:" + str(count4))
        print("all:" + str(count_all))


if __name__ == "__main__":
    MFCNInterface()
