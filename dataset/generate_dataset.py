from __future__ import print_function, division
import os
import re
import math
import random
import numpy as np
from scipy import misc, ndimage
from ImageAugmenter import ImageAugmenter

random.seed(42)

MAIN_DIR = os.path.dirname(os.path.realpath(__file__))

# change this dir to your filepath
DIRS = ["/media/aj/grab/ml/datasets/flickr-sky/orig"]

SCALE_HEIGHT = 32
SCALE_WIDTH = 64
RATIO_WIDTH_TO_HEIGHT = 2
EPSILON = 0.1
WRITE_TO = os.path.join(MAIN_DIR, "out_aug_32x64")

AUGMENTATIONS = 10
PADDING = 20

def main():
    fps_img = get_all_filepaths(DIRS)
    
    print("Augmenting images...")
    nb_written = 0
    for fp_idx, fp_img in enumerate(fps_img):
        print("Image %d of %d (%.2f%%) (%s)" % (fp_idx+1, len(fps_img), 100*(fp_idx+1)/len(fps_img), fp_img))
        try:
            filename = fp_img[fp_img.rfind("/")+1:]
            image = ndimage.imread(fp_img, mode="RGB")
            image_orig = np.copy(image)
            #misc.imshow(image)
            #print(image)
            #print(image.shape)
            
            height = image_orig.shape[0]
            width = image_orig.shape[1]
            wh_ratio = width / height
            batch = np.zeros((AUGMENTATIONS, height+(2*PADDING), width+(2*PADDING), 3), dtype=np.uint8)
            
            img_padded = np.pad(image, ((PADDING, PADDING), (PADDING, PADDING), (0, 0)), mode="median")
            for i in range(0, AUGMENTATIONS):
                batch[i] = np.copy(img_padded)
            
            ia = ImageAugmenter(width+(2*PADDING), height+(2*PADDING), channel_is_first_axis=False,
                            hflip=True, vflip=False,
                            scale_to_percent=(1.05, 1.2), scale_axis_equally=True,
                            rotation_deg=5, shear_deg=1,
                            translation_x_px=15, translation_y_px=15)
            batch = ia.augment_batch(batch)
            
            for i in range(0, AUGMENTATIONS):
                image = batch[i, PADDING:-PADDING, PADDING:-PADDING, ...]
                
                removed = 0
                while not (wh_ratio - EPSILON <= RATIO_WIDTH_TO_HEIGHT <= wh_ratio + EPSILON):
                    if wh_ratio < RATIO_WIDTH_TO_HEIGHT:
                        # height value is too high
                        # remove more from top than from bottom, because we have sky images and
                        # hence much similar content at top and only a few rows of pixels with
                        # different content at the bottom
                        if removed % 4 != 0:
                            # remove one row at the top
                            image = image[1:height-0, :, ...]
                        else:
                            # remove one row at the bottom
                            image = image[0:height-1, :, ...]
                    else:
                        # width value is too high
                        if removed % 2 == 0:
                            # remove one column at the left
                            image = image[:, 1:width-0, ...]
                        else:
                            # remove one column at the right
                            image = image[:, 0:width-1, ...]
                    
                    height = image.shape[0]
                    width = image.shape[1]
                    wh_ratio = width / height
                    removed += 1
                
                resized = misc.imresize(image, (SCALE_HEIGHT, SCALE_WIDTH))
                misc.imsave(os.path.join(WRITE_TO, filename.replace(".jp", "__%d.jp" % (i))), resized)
                nb_written += 1
        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
    print("Written %d" % (nb_written))
    print("Finished.")

def get_all_filepaths(fp_dirs):
    result_img = []
    result_coords = []
    for fp_dir in fp_dirs:
        fps = [f for f in os.listdir(fp_dir) if os.path.isfile(os.path.join(fp_dir, f))]
        fps = [os.path.join(fp_dir, f) for f in fps]
        fps_img = [fp for fp in fps if re.match(r".*\.(?:jpg|jpeg|png)$", fp)]
        result_img.extend(fps_img)

    return result_img

if __name__ == "__main__":
    main()
