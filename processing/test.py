import openslide
import multiresolutionimageinterface as mir
import numpy as np
from PIL import Image
import os

mask_url = '/home/suidong/Documents/camelyon17_data_backup/mask(copy)/'
slide_url = '/home/suidong/Documents/camelyon17_data_backup/slide/'
level = 9
is_bad = False
#s = openslide.OpenSlide('/home/suidong/Documents/camelyon16/Testset/Images/Test_001.tif')
#m = openslide.OpenSlide('/home/suidong/Documents/camelyon16/Testset/Ground_Truth/Masks/Test_001_Mask.tif')
#s = openslide.OpenSlide('/home/suidong/Documents/camelyon16/TrainingData/Train_Tumor/Tumor_001.tif')
#m = openslide.OpenSlide('/home/suidong/Documents/camelyon16/TrainingData/Ground_Truth/Mask/Tumor_001_Mask.tif')
#s = openslide.OpenSlide('/home/suidong/Documents/camelyon17_data_backup/slide/patient_000_node_0.tif')
#m = openslide.OpenSlide('/home/suidong/Documents/camelyon17_data_backup/test/test.tif')
#for i in range(level):
#    print(str(i) + 'level(scale,s,m):')
#    print(s.level_downsamples[i])
#    print(s.level_dimensions[i])
#    print(m.level_dimensions[i])
# scale = s.level_downsamples[level]
# split_x = range(0, 1344, 128)
# split_y = range(0, 1400, 128)
# tile = s.read_region((int(split_x[10]*scale), int(split_y[10]*scale)), level, (128, 128))
# tile.show()
for root, dirs, files in os.walk(mask_url):
    for filename in files:
        reader = mir.MultiResolutionImageReader()
        mr_image = reader.open(os.path.join(slide_url, filename))# type: mir.MultiResolutionImage
        mask_image = reader.open(os.path.join(mask_url, filename))# type: mir.MultiResolutionImage
        if mr_image is None:
            print("nofile:" + filename)
            continue
        is_bad = False
        for i in range(level):
            # print mr_image.getLevelDownsample(i)
            # print mask_image.getLevelDownsample(i)
            slide_d = mr_image.getLevelDimensions(i)
            mask_d = mask_image.getLevelDimensions(i)
            if slide_d != mask_d:
                print("bad:" + filename)
                is_bad = True
                break
        if is_bad is not True:
            print(filename)
