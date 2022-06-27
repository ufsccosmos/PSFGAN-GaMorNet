import time
import argparse
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
from astropy.io import fits

from data import *
from model import CGAN

def prepocess_train(img, cond):
    # img = scipy.misc.imresize(img, [conf.adjust_size, conf.adjust_size])
    # cond = scipy.misc.imresize(cond, [conf.adjust_size, conf.adjust_size])
    # h1 = int(np.ceil(np.random.uniform(1e-2, conf.adjust_size - conf.train_size)))
    # w1 = int(np.ceil(np.random.uniform(1e-2, conf.adjust_size - conf.adjust_size)))
    # img = img[h1:h1 + conf.train_size, w1:w1 + conf.train_size]
    # cond = cond[h1:h1 + conf.train_size, w1:w1 + conf.train_size]

    if np.random.random() > 0.5:
        img = np.fliplr(img)
        cond = np.fliplr(cond)

    img = img.reshape(1, conf.train_size, conf.train_size, conf.img_channel)
    cond = cond.reshape(1, conf.train_size, conf.train_size, conf.img_channel)
    return img, cond


def prepocess_test(img, cond):
    # img = scipy.misc.imresize(img, [conf.train_size, conf.train_size])
    # cond = scipy.misc.imresize(cond, [conf.train_size, conf.train_size])
    img = img.reshape(1, conf.train_size, conf.train_size, conf.img_channel)
    cond = cond.reshape(1, conf.train_size, conf.train_size, conf.img_channel)
    # img = img/127.5 - 1.
    # cond = cond/127.5 - 1.
    return img, cond


def test(mode):
    data = load_data()
    model = CGAN()

    saver = tf.train.Saver()

    counter = 0
    start_time = time.time()
    out_dirs = conf.result_paths
    filters_string = conf.filters_
    num_filters = conf.num_filters
    for f_index in range(num_filters):
        if not os.path.exists(conf.save_paths[f_index]):
            os.makedirs(conf.save_paths[f_index])
        if not os.path.exists(out_dirs[f_index]):
            os.makedirs(out_dirs[f_index])

    start_epoch = 0
    with tf.Session() as sess:    
        saver.restore(sess, conf.model_path)
        for epoch in xrange(start_epoch, conf.max_epoch):
            if (epoch + 1) == conf.test_epoch:
            # if (epoch + 1) % conf.save_per_epoch == 0:
                test_data = data[str(mode)]()
                for img, cond, name in test_data:
                    #name = name.replace('-'+filter_string+'.npy', '')
                    pimg, pcond = prepocess_test(img, cond)
                    gen_img = sess.run(model.gen_img, feed_dict={model.image: pimg, model.cond: pcond})
                    gen_img = gen_img.reshape(gen_img.shape[1:])

                    for f_index in range(num_filters):
                        fits_recover = conf.unstretch(gen_img[:, :, f_index])
                        hdu = fits.PrimaryHDU(fits_recover)
                        save_dir = '%s/epoch_%s/fits_output' % (out_dirs[f_index], epoch + 1)
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        filename = '%s/%s-%s.fits' % (save_dir, name, filters_string[f_index])
                        if os.path.exists(filename):
                            os.remove(filename)
                        hdu.writeto(filename)


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",default="eval")
    args = parser.parse_args()
    mode = args.mode
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(conf.use_gpu)
    test(mode)
    end_time = time.time()
    print 'inference time: '
    print str(end_time-start_time)