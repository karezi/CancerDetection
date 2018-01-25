from keras.callbacks import EarlyStopping
from networks.SimpleFCN import *
from PIL import Image
import time
import glob
from pylab import *
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

IMG_SUFFIX = "png"
MAP_SUFFIX = "png"
DATASET_ROOT_URL = "/Users/karezi/Desktop/datasets/tumor/camelyon16trainingdata/h5/"
H5_PATH = os.path.join(DATASET_ROOT_URL, str(int(time.time())) + ".h5")
IMG_PATH = "/Users/karezi/Desktop/datasets/tumor/camelyon16trainingdata/tumor_square_test/"
MAP_PATH = "/Users/karezi/Desktop/datasets/tumor/camelyon16trainingdata/mask_square_test/"
TEST_PATH = "/Users/karezi/Desktop/datasets/tumor/camelyon16trainingdata/tumor_square/Tumor_012_1.png"
GEN_PATH = "/Users/karezi/Desktop/datasets/tumor/camelyon16trainingdata/h5/1516516565.h5"


def detection_interface():
    images_files = glob.glob(os.path.join(IMG_PATH, "*." + IMG_SUFFIX))
    images = np.array([array(Image.open(x).convert("L"))[..., newaxis] for x in images_files])
    images_gray = images / 255.0
    print images.shape
    maps_files = glob.glob(os.path.join(MAP_PATH, "*." + MAP_SUFFIX))
    maps = np.array([np.array(Image.open(x).convert("L"))[..., newaxis] for x in maps_files])
    maps_gray = maps / 255.0
    total_len = len(images_gray)
    test_len = int(total_len * 0.2)
    train_len = total_len - test_len
    model = SimpleFCN().get_model()
    trainx, trainy = images_gray[0:train_len], maps_gray[0:train_len]
    early_stopping = EarlyStopping(monitor='val_loss', patience=1)
    model.fit(trainx, trainy, batch_size=4, epochs=1, validation_split=0.2, callbacks=[early_stopping])
    model.save(H5_PATH)
    testx, testy = images[train_len:total_len], maps[train_len:total_len]
    print(model.evaluate(testx, testy, batch_size=32, verbose=1, sample_weight=None))


def test_interface():
    test_image = Image.open(TEST_PATH).convert("L")
    predict_target = np.array([array(test_image)[..., newaxis]]) / 255.0
    model = SimpleFCN().get_model()
    model.load_weights(GEN_PATH)
    res = model.predict(predict_target)
    res_image = res[0][..., 0] * 255.0
    plt.figure()
    plt.imshow(res_image, cmap=cm.gray_r)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    # detection_interface()
    test_interface()
