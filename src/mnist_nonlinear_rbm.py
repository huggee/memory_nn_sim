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

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

args = sys.argv

print datetime.datetime.now()

n_visible  = int(args[1])
n_hidden   = int(args[2])
eta        = float(args[3])
sign_const = int(args[4])
image_size = int(np.sqrt(n_visible))
print(image_size, n_visible, n_hidden)

if sign_const == 1:
    vg_max = -0.9
    vg_min = -1.4
else:
    vg_max = -0.9
    vg_min = -1.4

n_batch        = 1
max_itr        = 100000
interval_print = 100
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
img = np.where(img > 63.0, 1.0, 0.0)
img = np.reshape(img, [items, n_visible])

### weight
vg        = np.random.uniform(vg_min, vg_max, [2, n_visible, n_hidden])
_vg       = np.zeros([2, n_visible, n_hidden])
delta_vg  = np.zeros([n_visible, n_hidden])
_delta_vg = np.zeros([n_visible, n_hidden])
grad      = np.zeros([n_visible, n_hidden])
_grad     = np.zeros([n_visible, n_hidden])
ada_grad  = np.zeros([2, n_visible, n_hidden])

bias_v = np.zeros([n_visible])
bias_h = np.zeros([n_hidden])
grad_bv = np.zeros([n_visible])
grad_bh = np.zeros([n_hidden])
delta_bv = np.zeros([n_visible])
delta_bh = np.zeros([n_hidden])


### sign info
vg_sign  = np.random.choice([True, False], size = [n_visible, n_hidden])
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

    ### encode
    current_encode = np.zeros([2, n_hidden])
    for i in range(2):
        if sign_const == 1:
            a = 1.4 * np.exp(16.0 * vg[i]) * vg_sign[i]
        else:
            a = 1.4 * np.exp(16.0 * vg[i])
        b = 1.0 - np.exp(-0.005 * x_input)
        current_encode[i] = np.matmul(b, a)

    hidden_x = current_encode[0] - current_encode[1] + bias_h
    hidden_y = np.where(hidden_x > 0, 1.0, 0.0)
    hidden_prob = sigmoid(hidden_x)
    prob_max = np.max(hidden_prob)
    hidden_state = np.where(hidden_prob >= np.random.rand() * prob_max, 1.0, 0.0)

    ### decode
    current_decode = np.zeros([2, n_visible])
    for i in range(2):
        if sign_const == 1:
            a = 1.4 * np.exp(16.0 * np.transpose(vg[i])) * np.transpose(vg_sign[i])
        else:
            a = 1.4 * np.exp(16.0 * np.transpose(vg[i]))
        b = 1.0 - np.exp(-0.005 * -hidden_state)
        current_decode[i] = np.matmul(b, a)

    output_x = current_decode[0] - current_decode[1] + bias_v
    output_y = np.where(output_x > 0, 1.0, 0.0)
    output_y = 1.0 / (1.0 + np.exp(-output_x))
    output_prob = sigmoid(output_x)
    prob_max = np.max(output_prob)
    output_state = np.where(output_prob >= np.random.rand() * prob_max, 1.0, 0.0)

    current_e2 = np.zeros([2, n_hidden])
    for i in range(2):
        if sign_const == 1:
            a = 1.4 * np.exp(16.0 * vg[i]) * vg_sign[i]
        else:
            a = 1.4 * np.exp(16.0 * vg[i])
        b = 1.0 - np.exp(-0.005 * output_state)
        current_e2[i] = np.matmul(b, a)
    hidden_e2 = current_e2[0] - current_e2[1] + bias_h
    hidden_prob_e2 = sigmoid(hidden_e2)

    d_output = output_y
    # d_output = output_y * -1.0 + 1.0

    ### accuracy
    # a        = np.where(output_y > 0.0, 1.0, 0.0)
    a        = output_state
    eq       = np.equal(a, x_input)
    avg_eq  += np.sum(eq)
    avg_acc += float(np.sum(eq)) / float(n_visible)

    ### loss calculation
    loss     = x_input - output_y
    avg_err += np.mean(np.power(loss, 2))

    ### delta
    # a = np.reshape(loss, [n_visible, 1])
    a = np.reshape(loss * d_output, [n_visible, 1])
    b = np.reshape(hidden_y, [1, n_hidden])
    grad = np.matmul(a, b)
    # delta_vg += grad

    v_pos = np.reshape(x_input, [-1, 1])
    p_pos = np.reshape(hidden_prob, [1, -1])
    v_neg = np.reshape(output_state, [-1, 1])
    p_neg = np.reshape(hidden_prob_e2, [1, -1])
    grad = np.matmul(v_pos, p_pos) - np.matmul(v_neg, p_neg)
    # pdb.set_trace()

    grad_bv = x_input - output_state
    grad_bh = hidden_prob - hidden_prob_e2

    delta_bv += eta * grad_bv
    delta_bh += eta * grad_bh

    # #++ gradient descent +++#
    # delta_vg += eta * grad

    #++ momentum ++
    delta_vg += eta * grad + 0.9 * _delta_vg
    _delta_vg = eta * grad

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
        delta_bv = delta_bv / n_batch
        delta_bh = delta_bh / n_batch

        ### sign constraint
        if sign_const == 1:
            delta_vg_2 = np.array([delta_vg, -delta_vg]) * vg_sign
        else:
            delta_vg_2 = np.array([delta_vg, -delta_vg])

        vg_new = vg + delta_vg_2
        bv_new = bias_v + delta_bv
        bh_new = bias_h + delta_bh

        ### max/min constraint
        if sign_const == 1:
            vg_new   = np.where((vg_new != 0.0) & (vg_new > vg_max), vg_max, vg_new)
            vg_new   = np.where((vg_new != 0.0) & (vg_new < vg_min), vg_min, vg_new)
        else:
            vg_new   = np.where(vg_new > vg_max, vg_max, vg_new)
            vg_new   = np.where(vg_new < vg_min, vg_min, vg_new)


        ### update/reset
        # _delta_vg = delta_vg_2
        delta_vg  = np.zeros([n_visible, n_hidden])
        delta_bv = np.zeros([n_visible])
        delta_bh = np.zeros([n_hidden])
        vg = vg_new
        bias_v = bv_new
        bias_h = bh_new
        epoch += 1

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

