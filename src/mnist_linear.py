# import matplotlib.pyplot as plt
import datetime
import gzip
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import shutil
import struct
import sys
from PIL import Image
from six.moves import urllib

args = sys.argv

n_visible  = int(args[1])
n_hidden   = int(args[2])
sign_const = int(args[3])
image_size = int(np.sqrt(n_visible))
print(image_size, n_visible, n_hidden)

eta    = 0.01
vg_max = 1.0
vg_min = -1.0

n_batch        = 1
max_itr        = 10000
interval_print = 1
epoch          = 0

avg_err = 0.0
avg_acc = 0.0
avg_eq  = 0.0

### get mnist dataset
data_dir  = '../mnist_data/'
file_name = 'train-images-idx3-ubyte.gz'
file_path = data_dir + file_name
src_url   = 'https://storage.googleapis.com/cvdf-datasets/mnist/' + file_name

if not os.path.exists(data_dir):
    os.mkdir(data_dir)

if not os.path.exists(file_path):
    temp_file_path, _ = urllib.request.urlretrieve(src_url)
    shutil.copy(temp_file_path, file_path)

### extract mnist images
dtype = np.dtype(np.uint32).newbyteorder('>')
with gzip.open(file_path, 'rb') as f:
    magic_number = np.frombuffer(f.read(4), dtype = dtype)[0]
    if magic_number != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s'\
                             % (magic_number, file_path))
    items  = np.frombuffer(f.read(4), dtype = dtype)[0]
    rows   = np.frombuffer(f.read(4), dtype = dtype)[0]
    cols   = np.frombuffer(f.read(4), dtype = dtype)[0]
    buf    = f.read(items * rows * cols)
    images = np.frombuffer(buf, dtype = np.uint8)
    images = np.reshape(images, [items, rows, cols])

### normalize and resize
img = [Image.fromarray(i) for i in images]
img = [i.resize((image_size, image_size)) for i in img]
img = [np.asarray(i) for i in img]
img = np.array(img)
img = np.where(img > 63.0, 1.0, -1.0)
img = np.reshape(img, [items, n_visible])

### weight
# vg        = np.random.uniform(vg_min, vg_max, [n_visible, n_hidden])
vg        = np.random.normal(size = [n_visible, n_hidden])
_vg       = np.zeros([n_visible, n_hidden])
delta_vg  = np.zeros([n_visible, n_hidden])
_delta_vg = np.zeros([n_visible, n_hidden])
grad      = np.zeros([n_visible, n_hidden])
_grad     = np.zeros([n_visible, n_hidden])

### sign info
# vg_sign  = np.greater_equal(vg[0], vg[1])
vg_sign  = np.random.choice([True, False], size = [n_visible, n_hidden])
vg_sign  = np.array([vg_sign, ~vg_sign])

### sign constraint
if sign_const == 1:
    vg = vg * vg_sign


for iteration in range(max_itr):
    x_input = img[np.random.randint(0, items)]


    ### encode
    hidden_x = np.matmul(x_input, vg)
    # hidden_y = hidden_x * 1e8
    hidden_y   = np.tanh(hidden_x)
    # hidden_y   = np.where(hidden_x > 0, 1.0, -1.0)

    ### decode
    output_x = np.matmul(hidden_y, np.transpose(vg))
    # output_y = output_x * 1e8
    output_y = np.tanh(output_x)
    # output_y = np.where(output_x > 0, 1.0, -1.0)

    # d_output = output_y
    d_output = 1.0 / np.power(np.cosh(output_x), 2)

    ### loss calculation
    loss     = x_input - output_y
    avg_err += np.mean(np.power(loss, 2))

    ### accuracy
    a        = np.where(output_y > 0.0, 1.0, -1.0)
    eq       = np.equal(a, x_input)
    avg_eq  += np.sum(eq)
    avg_acc += float(np.sum(eq)) / float(n_visible)

    ### delta
    # a = np.reshape(loss, [n_visible, 1])
    a = np.reshape(loss * d_output, [n_visible, 1])
    b = np.reshape(hidden_y, [1, n_hidden])
    grad = np.matmul(a, b)
    # delta_vg += grad

    #++ gradient descent +++#
    delta_vg += eta * grad

    # #++ memontum ++
    # delta_vg += eta * grad + 0.9 * _delta_vg
    # _delta_vg = eta * grad

    # #++ adagrad ++
    # _grad += grad * grad
    # delta_vg += eta * grad / (np.sqrt(_grad) + 1e-8)

    # #++ rmsprop ++
    # alpha = 0.9
    # _grad = alpha * _grad + (1.0 - alpha) * grad * grad
    # delta_vg += eta * grad / (np.sqrt(_grad) + 1e-8)


    ### train
    if iteration % n_batch == 0:
        delta_vg = delta_vg / n_batch
        # if sign_const == 1:
        #     delta_vg = delta_vg * vg_sign

        vg_new = vg + delta_vg

        ### max/min constraint
        # if sign_const == 1:
        #     vg_new   = np.where((vg_new != 0.0) & (vg_new > vg_max), vg_max, vg_new)
        #     vg_new   = np.where((vg_new != 0.0) & (vg_new < vg_min), vg_min, vg_new)
        # else:
        #     vg_new   = np.where(vg_new > vg_max, vg_max, vg_new)
        #     vg_new   = np.where(vg_new < vg_min, vg_min, vg_new)

        # vg_new   = np.where(vg_new > vg_max, vg_max, vg_new)
        # vg_new   = np.where(vg_new < vg_min, vg_min, vg_new)

        # ### sign constraint
        # if sign_const == 1:
        #     _vg_sign = np.greater_equal(vg_new[0], vg_new[1])
        #     sign_xor = vg_sign ^ _vg_sign
        #     sign_xor = np.array([sign_xor, sign_xor])
        #     vg_new   = vg * sign_xor + vg_new * ~sign_xor

        ### update/reset
        # _delta_vg = delta_vg_2
        delta_vg  = np.zeros([n_visible, n_hidden])
        vg        = vg_new
        epoch    += 1

    ### display
    if iteration % interval_print == 0:
        avg_err /= interval_print
        avg_acc /= interval_print
        avg_eq  /= interval_print
        print('%d, %4d, %f, %f, %2.1f/%d' % (epoch - 1,\
                                             iteration,\
                                             avg_err,\
                                             avg_acc,\
                                             avg_eq,\
                                             n_visible\
                                             ))
        avg_err = 0.0
        avg_acc = 0.0
        avg_eq  = 0.0

        # pdb.set_trace()

# n = 10
# plt.figure(figsize = (20, 4))
# for i in range(n):
#     # ax = plt.subplot(2, n, i + 1)
#     # plt.imshow(x_set[i].reshape(28, 28))
#     # plt.gray()
#     # ax.get_xaxis().set_visible(False)
#     # ax.get_yaxis().set_visible(False)
# 
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(img[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.savefig('mnist_ae_nol_pm.png')

