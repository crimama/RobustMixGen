import numpy as np

def ro_mixgen(image, text, dataset:str):
    ...

def ro_mixgen_batch(image, text, dataset:str):
    batch_size = image.size()[0]
    index = np.random.permutation(batch_size)