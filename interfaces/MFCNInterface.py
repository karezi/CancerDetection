import os
from PIL import Image
import glob
from pylab import *
import numpy as np
from networks.ColorFCN import *
import time
from keras.callbacks import EarlyStopping
from postprocessing.LossHistory import *

LEVEL = 6
BATCH_SIZE = 2
EPOCHS = 100
INPUT_ROOT = '/home/suidong/Documents/output/camelyon17/'
LOG_ROOT = '/home/suidong/Documents/output/camelyon17/log/'
SLIDE_TRAIN_DIR = INPUT_ROOT + 'slide_train/' + str(LEVEL) + '/'
SLIDE_TEST_DIR = INPUT_ROOT + 'slide_test/' + str(LEVEL) + '/'
MASK_TRAIN_DIR = INPUT_ROOT + 'mask_train/' + str(LEVEL) + '/'
MASK_TEST_DIR = INPUT_ROOT + 'mask_test/' + str(LEVEL) + '/'
H5_PATH = os.path.join(LOG_ROOT, "MFCNInterface." + str(LEVEL) + "." + str(int(time.time())) + ".h5")


class MFCNInterface(object):
    def __init__(self):
        self.train_data()

    def train_data(self):
        file_list = glob.glob(os.path.join(SLIDE_TRAIN_DIR, '*'))
        images = np.array([array(Image.open(x)) for x in file_list])
        print(images.shape)
        mask_list = glob.glob(os.path.join(MASK_TRAIN_DIR, '*'))
        masks = np.array([array(Image.open(x))[..., newaxis] for x in mask_list])
        print(masks.shape)
        total_len = images.shape[0]
        train_len = int(total_len * 0.8)
        model = ColorFCN().get_model()
        train_x, train_y = images[0:train_len], masks[0:train_len]
        test_x, test_y = images[train_len:total_len], masks[train_len:total_len]
        early_stopping = EarlyStopping(monitor='val_loss', patience=1)
        # model.fit(train_x, train_y, batch_size=4, epochs=1, validation_split=0.2, callbacks=[early_stopping])
        history = LossHistory()
        model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(test_x, test_y), callbacks=[history])
        model.save(H5_PATH)
        score = model.evaluate(test_x, test_y, batch_size=32, verbose=1, sample_weight=None)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        history.loss_plot('epoch')


if __name__ == "__main__":
    MFCNInterface()
