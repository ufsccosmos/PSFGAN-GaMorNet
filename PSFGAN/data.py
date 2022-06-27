import os
import glob
import numpy as np

from config import Config as conf


#def load(path):
#    print(path)
#    for i in os.listdir(path):
#        all = np.load(path + '/' + i)
#        img, cond = all[:, :conf.img_size], all[:, conf.img_size:]
#        yield (img, cond, i)

def load(paths, mode):
    ## Print out data paths
    #for f_index in range(conf.num_filters):
    #    print(paths[f_index])
    # Find out fits image ids using the first filter
    npy_path = '%s/%s/*-%s.npy' % (paths[0], mode, conf.filters_[0])
    files = glob.glob(npy_path)
    image_ids = []
    for i in files:
        image_id = os.path.basename(i).replace('-' + conf.filters_[0] + '.npy', '')
        image_ids.append(image_id)
    num_images = len(image_ids)
    # Then read data from all filters according to image_ids
    for image_id in image_ids:
        img = np.zeros([conf.img_size, conf.img_size, conf.img_channel])
        cond = np.zeros([conf.img_size, conf.img_size, conf.img_channel])
        for f_index in range(conf.num_filters):
            all_per_filter = np.load('%s/%s/%s-%s.npy' % (paths[f_index], mode, image_id, conf.filters_[f_index]))
            img[:, :, f_index] = all_per_filter[:, :conf.img_size, 0]
            cond[:, :, f_index] = all_per_filter[:, conf.img_size:, 0]
        yield (img, cond, image_id)

def load_data(paths=conf.data_paths):
    data = dict()
    data["test"] = lambda: load(paths, 'test')
    data["train"] = lambda: load(paths, 'train')
    data["eval"] = lambda: load(paths, 'eval')
    data["gmn_train"] = lambda: load(paths, 'gmn_train')
    data["gmn_eval"] = lambda: load(paths, 'gmn_eval')
    return data
