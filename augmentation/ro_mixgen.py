import numpy as np
import pycocotools as COCO
import os

def ro_mixgen(image, text, threshold=0.01):
    ...
    # check json file

def ro_mixgen_batch(image, text):
    batch_size = image.size()[0]
    index = np.random.permutation(batch_size)