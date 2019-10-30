import numpy as np
import scipy.misc
import os
import urllib
import gzip
import pickle
from glob import glob
from scipy.misc import imsave, imresize, imread
# import moviepy.editor as mpy

def rand_clip(x, seq_length):
    start = np.random.randint(x.shape[0] - seq_length + 1)
    #print('start:',start)
    #start = 0
    return x[start:start+seq_length]

def chair_generator(batch_size, seq_length, data, size):
    global start
    #start = 0
    def get_epoch():
        global start
        if seq_length == 1:
            data_all = data.reshape((-1, size*size*3))
        elif seq_length == 31:
            data_all = data.reshape((-1, 31, size*size*3))
        elif seq_length == 4:
            data_all = []
            for d in data:
                data_all.append(rand_clip(d, seq_length*2))
                #data_all.append(d[start:start+seq_length])
            data_all = np.asarray(data_all)
            print(data_all.shape)
            data_all = data_all[:,:seq_length*2,:]
            #print('start:',start)
            #start=start+4
        else:    
            data_all = data[:,:seq_length,:]
                
        np.random.shuffle(data_all)
        #print 'data_shape', data_all.shape
        for i in range(data_all.shape[0] // batch_size):
            yield data_all[i*batch_size:(i+1)*batch_size]    
    return get_epoch

def load(seq_length, batch_size, size, data_dir, num_dev=10):
    data = np.load(os.path.join(data_dir, 'sar_'+str(size)+'.npy'))
    print(data.shape)
    data = np.transpose(data, [0, 1, 4, 2, 3])
    data = data.reshape((-1, 31, size*size*3))
    np.random.shuffle(data)
    print('623862183',data[num_dev:])
    return (
            chair_generator(batch_size, seq_length, data[num_dev:], size),
            chair_generator(batch_size, seq_length, data[:num_dev], size)
    )

# def npy_to_gif(npy, size):
#     for i in xrange(npy.shape[0]):
#         clip = mpy.ImageSequenceClip(list(npy[i]), fps=5)
#         clip.write_gif(str(size)+'_'+str(i)+'.gif')

def npy_to_image(npy, size):
    for i in range(npy.shape[0]):
        imsave(str(size)+'_'+str(i)+'.png', npy[i])

def imread(path, grayscale=False):
    if (grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def center_crop(image, size):
    #print(image.shape)
    image = image[:,:,:]
    print(image.shape)
    image = imresize(image, (size, size))
    print(image.shape)
    return np.array(image)

def get_image(image_path, size, grayscale):
    image = imread(image_path, grayscale)
    return center_crop(image, size)

def print_array(x):
    print (x.shape, x.dtype, x.max(), x.min())

def convert_to_numpy(size):
    data = glob(os.path.join('../data/*/*.jpg'))
    data.sort()
    print (data)
    sample = [get_image(d, size, grayscale=False) for d in data]
    sample_inputs = np.array(sample).astype(np.int32)
    print(sample_inputs.shape)
    #npy_to_image(sample_inputs, size)
    sample_inputs = sample_inputs.reshape((-1, 31, size, size, 3))
    #sample_inputs = np.transpose(sample_inputs, [1, 0, 2, 3, 4])
    print_array(sample_inputs)
    #npy_to_gif(sample_inputs, size)
    np.save('sar_'+str(size), sample_inputs)

#convert_to_numpy(64)
#convert_to_numpy(32)
