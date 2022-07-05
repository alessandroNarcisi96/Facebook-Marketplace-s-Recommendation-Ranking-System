from PIL import Image
import numpy as np
import os
import splitfolders

def clean(path: str,dest: str,ds):
    dirs = os.listdir(path)
    final_size = 256
    for n, item in enumerate(dirs, 1):
        im = Image.open(path + item)
        new_im = __resize_image(final_size, im)
        id = item.replace('.jpg', '')
        imm = ds[ds['image_id']==id]
        if imm.shape[0] == 1 :
            new_im.save(dest+item)

def __resize_image(final_size, im):
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im

def convertImageToArray(path: str):
    images = []
    dirs = os.listdir(path)
    for n, item in enumerate(dirs, 1):
        im = Image.open(path + item)       
        id = item.split("_")[1]
        images.append([np.array(im).flatten(),id.replace('.jpg', '')])
    return images
    
def split_images(train_size: float,val_size: float,test_size: float,path: str,output_path: str):
    splitfolders.ratio(path, output=output_path, seed=1337, ratio=(train_size, val_size,test_size)) 