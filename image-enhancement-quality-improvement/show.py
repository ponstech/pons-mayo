"""
This function sets specifc setting for how the image is displayed
"""
import numpy as np
import matplotlib.pyplot as plt



def show(img, cmap='gray', save=False):
    fig = plt.figure(figsize=(img.shape[1] / plt.rcParams['figure.dpi'], img.shape[0] / plt.rcParams['figure.dpi']), dpi=plt.rcParams['figure.dpi'])
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.tight_layout(pad=0)
    if save:
        plt.savefig('filename.png', bbox_inches='tight', pad_inches=0)
    plt.show()
