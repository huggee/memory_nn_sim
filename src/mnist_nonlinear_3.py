import matplotlib.pyplot as plt
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

print datetime.datetime.now()

n_visible  = int(args[1])
n_hidden   = int(args[2])
sign_const = int(args[3])
image_size = int(np.sqrt(n_visible))
print(image_size, n_visible, n_hidden)

if sign_const == 1:
    eta    = 0.001
    vg_max = -0.9
    vg_min = -1.4
else:
    eta    = 0.001
    vg_max = -0.9
    vg_min = -1.4

n_batch        = 1
max_itr        = 10000
interval_print = 10
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
vg        = np.random.uniform(vg_min, vg_max, [2, n_visible, n_hidden])
# vg[0]     = np.random.uniform(vg_min, vg_max, [n_visible, n_hidden])
# vg[1]     = np.random.uniform(vg_min-0.2, vg_max+0.2, [n_visible, n_hidden])
_vg       = np.zeros([2, n_visible, n_hidden])
delta_vg  = np.zeros([2, n_visible, n_hidden])
_delta_vg = np.zeros([n_visible, n_hidden])
grad      = np.zeros([2, n_visible, n_hidden])
_grad     = np.zeros([n_visible, n_hidden])
ada_grad  = np.zeros([2, n_visible, n_hidden])

### sign info
vg_sign  = np.random.choice([True, False], size = [n_visible, n_hidden])
# vg_sign  = np.random.choice([True, False], size = [n_visible, n_hidden], p = [0.9, 0.1])
vg_sign  = np.array([vg_sign, ~vg_sign])

### sign constraint
if sign_const == 1:
    vg = vg * vg_sign

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


for iteration in range(max_itr):
    x_input = img[np.random.randint(0, items)]
    x_p = np.where(x_input == 1.0, 1.0, 0.0)
    x_m = np.where(x_input == -1.0, 1.0, 0.0)
    x_pm = np.array([x_p, x_m])

    ### encode
    current_encode = np.zeros([2, n_hidden])
    for i in range(2):
        a = 1.4 * np.exp(16.0 * vg[i])
        b = 1 - np.exp(-0.005 * x_pm[i])
        current_encode[i] = np.matmul(b, a)

    hidden_x = current_encode[0] - current_encode[1]
    hidden_y = np.where(hidden_x > 0, 1.0, -1.0)
    h_p = np.where(hidden_y == 1.0, 1.0, 0.0)
    h_m = np.where(hidden_y == -1.0, 1.0, 0.0)
    h_pm = np.where([h_p, h_m])

    ### decode
    current_decode = np.zeros([2, n_visible])
    for i in range(2):
        a = 1.4 * np.exp(16.0 * np.transpose(vg[i]))
        b = 1 - np.exp(-0.005 * -h_pm[i])
        current_decode[i] = np.matmul(b, a)

    output_x = current_decode[0] - current_decode[1]
    output_y = np.where(output_x > 0, 1.0, -1.0)
    y_p = np.where(output_y == 1.0, 1.0, 0.0)
    y_m = np.where(output_y == -1.0, 1.0, 0.0)
    y_pm = np.where([y_p, y_m])

    d_output = output_y
    # d_output = 1.0 / np.power(np.cosh(output_x * 1e7), 2)

    ### loss calculation
    # loss     = x_input - output_y
    loss = x_pm - y_pm
    avg_err += np.mean(np.power(loss, 2))

    ### accuracy
    a        = np.where(output_y > 0.0, 1.0, -1.0)
    eq       = np.equal(a, x_input)
    avg_eq  += np.sum(eq)
    avg_acc += float(np.sum(eq)) / float(n_visible)

    ### delta
    # a = np.reshape(loss * d_output, [n_visible, 1])
    # b = np.reshape(hidden_y, [1, n_hidden])
    for i in range(2):
        a = np.reshape(loss[i], [n_visible, 1])
        b = np.reshape(h_pm[i], [1, n_hidden])
        grad[i] = np.matmul(a, b)
    # grad = np.matmul(a, b)
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

        ### sign constraint
        if sign_const == 1:
            # delta_vg_2 = np.array([delta_vg, -delta_vg]) * vg_sign
            delta_vg_2 = delta_vg
        else:
            # delta_vg_2 = np.array([delta_vg, -delta_vg])
            delta_vg_2 = delta_vg

        vg_new = vg - delta_vg_2

        ### max/min constraint
        if sign_const == 1:
            # vg_new   = np.where(vg_new > vg_max, vg_max, vg_new)
            # vg_new   = np.where(vg_new < vg_min, vg_min, vg_new)
            # vg_new   = vg_new * vg_sign
            # vg_new   = np.where((vg_new > vg_max, 0, vg_new)
            # vg_new   = np.where((vg_new < vg_min, 0, vg_new)
            vg_new   = np.where((vg_new == 0.0) | (vg_new > vg_max), 0, vg_new)
            # vg_new   = np.where((vg_new == 0.0) | (vg_new < vg_min), 0, vg_new)
            # vg_new   = np.where((vg_new != 0.0) & (vg_new > vg_max), vg_max, vg_new)
            vg_new   = np.where((vg_new != 0.0) & (vg_new < vg_min), vg_min, vg_new)
            # vg_new   = np.where((vg_new != 0.0) & (vg_new > vg_max), (vg_min + vg_max) / 2.0, vg_new)
            # vg_new   = np.where((vg_new != 0.0) & (vg_new < vg_min), (vg_max + vg_min) / 2.0, vg_new)
            # vg_new[0] = np.where((vg_new[0] != 0.0) & (vg_new[0] > (vg_max)), vg_max, vg_new[0])
            # vg_new[0] = np.where((vg_new[0] != 0.0) & (vg_new[0] < (vg_min)), vg_min, vg_new[0])
            # vg_new[1] = np.where((vg_new[1] != 0.0) & (vg_new[1] > (vg_max+0.2)), vg_max+0.2, vg_new[1])
            # vg_new[1] = np.where((vg_new[1] != 0.0) & (vg_new[1] < (vg_min-0.2)), vg_min-0.2, vg_new[1])
        else:
            vg_new   = np.where(vg_new > vg_max, vg_max, vg_new)
            vg_new   = np.where(vg_new < vg_min, vg_min, vg_new)

        ### update/reset
        # _delta_vg = delta_vg_2
        delta_vg  = np.zeros([2, n_visible, n_hidden])
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

np.save('vg.npy', vg)
# vg0 = np.reshape(vg[0], -1)
# vg0 = np.delete(vg0, np.where(vg0 == 0.0))
# plt.hist(vg0, bins = 100)
# plt.show()
pdb.set_trace()

