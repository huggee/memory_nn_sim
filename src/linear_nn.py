import matplotlib.pyplot as plt
import datetime
import gzip
import numpy as np
import os
import pdb
import shutil
import sys
from PIL import Image
from six.moves import urllib

args = sys.argv

now = datetime.datetime.now()
print now

### n_visible:  number of visible neurons 
### n_hidden:   number of hidden neurons 
### eta:        leraning rate
### sign_const: valid flag for sign constraint
n_visible  = int(args[1])
n_hidden   = int(args[2])
eta        = float(args[3])
sign_const = int(args[4])
image_size = int(np.sqrt(n_visible))
print(image_size, n_visible, n_hidden)

### max / min gate voltage
if sign_const == 1:
    vg_max = 1.0
    vg_min = 1e-3
else:
    vg_max = 1.0
    vg_min = 1e-3

### n_batch: number of data in a batch
### max_itr: max number of learning
### interval_print: interval to print log
### epoch: number of learning iteration up to now
n_batch        = 1
max_itr        = 10000
interval_print = 10
epoch          = 0

### avg_err: cost
### avg_acc: match rate
### avg_eq:  match number
avg_err = 0.0
avg_acc = 0.0
avg_eq  = 0.0

#########################
### get mnist dataset ###
#########################
data_dir  = '../mnist_data/'
file_name = 'train-images-idx3-ubyte.gz'
file_path = data_dir + file_name
src_url   = 'https://storage.googleapis.com/cvdf-datasets/mnist/' + file_name

if not os.path.exists(data_dir):
    os.mkdir(data_dir)

if not os.path.exists(file_path):
    temp_file_path, _ = urllib.request.urlretrieve(src_url)
    shutil.copy(temp_file_path, file_path)

############################
### extract mnist images ###
############################
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

############################
### normalize and resize ###
############################
img = [Image.fromarray(i) for i in images]
img = [i.resize((image_size, image_size)) for i in img]
img = [np.asarray(i) for i in img]
img = np.array(img)
img = np.where(img > 63.0, 1.0, 0.0)
img = np.reshape(img, [items, n_visible])

##################################
### initialize weight adn grad ###
##################################
vg        = np.random.uniform(vg_min, vg_max, [2, n_visible, n_hidden])
# vg = np.random.uniform(0, 1.0, [2, n_visible, n_hidden])
_vg       = np.zeros([2, n_visible, n_hidden])
delta_vg  = np.zeros([n_visible, n_hidden])
_delta_vg = np.zeros([n_visible, n_hidden])
grad      = np.zeros([n_visible, n_hidden])
_grad     = np.zeros([n_visible, n_hidden])
ada_grad  = np.zeros([2, n_visible, n_hidden])

### sign info
vg_sign  = np.random.choice([True, False], size = [n_visible, n_hidden])
vg_sign  = np.array([vg_sign, ~vg_sign])

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

with open('not_match.txt', mode = 'w') as f:
    f.write(str(now)+'\n')

#####################
### learning loop ###
#####################
f = open('not_match.txt', mode = 'a')
not_match_sum_en = 0.0
not_match_sum_de = 0.0
for iteration in range(max_itr):
    x_input = img[np.random.randint(0, items)]

    ##############
    ### encode ###
    ##############
    current_encode = np.zeros([2, n_hidden])
    volts_nonground = np.zeros([2, n_hidden])
    for i in range(2):
        if sign_const == 1:
            # a = 1.4 * np.exp(16.0 * vg[i]) * vg_sign[i]
            a = 1.0 / vg[i] * vg_sign[i]
        else:
            # a = 1.4 * np.exp(16.0 * vg[i])
            a = 1.0 / vg[i]
        # b = 1.0 - np.exp(-0.005 * x_input)
        b = x_input
        current_encode[i] = np.matmul(b, a)
        a_sum = np.sum(a, axis=0)
        volts_nonground[i] = current_encode[i] / a_sum

    comp_c = np.where(current_encode[0] > current_encode[1], 1.0, 0.0)
    comp_v = np.where(volts_nonground[0] > volts_nonground[1], 1.0, 0.0)
    not_match = np.where(comp_c != comp_v, 1.0, 0.0)
    not_match_sum_en += np.sum(not_match)
    not_match_len_en = not_match.shape[0]
    
    hidden_x = current_encode[0] - current_encode[1]
    hidden_y = np.where(hidden_x > 0, 1.0, 0.0)

    ##############
    ### decode ###
    ##############
    current_decode = np.zeros([2, n_visible])
    volts_nonground_de = np.zeros([2, n_visible])
    for i in range(2):
        if sign_const == 1:
            # a = 1.4 * np.exp(16.0 * np.transpose(vg[i])) * np.transpose(vg_sign[i])
            a = 1.0 / np.transpose(vg[i]) * np.transpose(vg_sign[i])
        else:
            # a = 1.4 * np.exp(16.0 * np.transpose(vg[i]))
            a = 1.0 / np.transpose(vg[i])
        # b = 1.0 - np.exp(-0.005 * -hidden_y)
        b = hidden_y
        current_decode[i] = np.matmul(b, a)
        a_sum = np.sum(a, axis=0)
        volts_nonground_de[i] = current_decode[i] / a_sum

    comp_c = np.where(current_decode[0] > current_decode[1], 1.0, 0.0)
    comp_v = np.where(volts_nonground_de[0] > volts_nonground_de[1], 1.0, 0.0)
    not_match = np.where(comp_c != comp_v, 1.0, 0.0)
    not_match_sum_de += np.sum(not_match)
    not_match_len_de = not_match.shape[0]

    output_x = current_decode[0] - current_decode[1]
    output_y = np.where(output_x > 0, 1.0, 0.0)
    d_output = output_y


    ################
    ### accuracy ###
    ################
    a        = np.where(output_y > 0.0, 1.0, 0.0)
    eq       = np.equal(a, x_input)
    avg_eq  += np.sum(eq)
    avg_acc += float(np.sum(eq)) / float(n_visible)

    ############
    ### loss ###
    ############
    loss     = x_input - output_y
    avg_err += np.mean(np.power(loss, 2))

    #############
    ### delta ###
    #############
    # a = np.reshape(loss, [n_visible, 1])
    # a = np.reshape(loss * d_output, [n_visible, 1])
    a = np.reshape(loss, [n_visible, 1])
    b = np.reshape(hidden_y, [1, n_hidden])
    grad = np.matmul(a, b)
    # delta_vg += grad

    ##################
    ### delta calc ###
    ##################
    ### gradient descent ###
    delta_vg += eta * grad

    # ### momentum ###
    # delta_vg += eta * grad + 0.9 * _delta_vg
    # _delta_vg = eta * grad

    # ### adagrad ###
    # _grad += grad * grad
    # delta_vg += eta * grad / (np.sqrt(_grad) + 1e-8)

    # ### rmsprop ###
    # alpha = 0.9
    # _grad = alpha * _grad + (1.0 - alpha) * grad * grad
    # delta_vg += eta * grad / (np.sqrt(_grad) + 1e-8)


    ##############
    ### update ###
    ##############
    if iteration % n_batch == 0:
        delta_vg = delta_vg / n_batch

        ### sign constraint
        if sign_const == 1:
            delta_vg_2 = np.array([-delta_vg, delta_vg]) * vg_sign
        else:
            delta_vg_2 = np.array([delta_vg, delta_vg])

        vg_new = vg + delta_vg_2

        ### max/min constraint
        if sign_const == 1:
            vg_new   = np.where(vg_new > vg_max, vg_max, vg_new)
            vg_new   = np.where(vg_new < vg_min, vg_min, vg_new)
        else:
            vg_new   = np.where(vg_new > vg_max, vg_max, vg_new)
            vg_new   = np.where(vg_new < vg_min, vg_min, vg_new)

        ### update/reset
        # _delta_vg = delta_vg_2
        delta_vg  = np.zeros([n_visible, n_hidden])
        vg        = vg_new
        epoch    += 1

    ### display
    if iteration % interval_print == 0:
        if iteration == 0:
            acc_length = 1
        else:
            acc_length = interval_print

        avg_err /= acc_length
        avg_acc /= acc_length
        avg_eq  /= acc_length
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
        not_match_sum_en = float(not_match_sum_en) / acc_length
        not_match_sum_de = float(not_match_sum_de) / acc_length
        f.write('%5d: \t %.1f/%d \t %.1f/%d \n' % \
                (iteration, not_match_sum_en, int(not_match_len_en), \
                not_match_sum_de, int(not_match_len_de)))
        not_match_sum = 0.0

np.save('vg.npy', vg)
# vg0 = np.reshape(vg[0], -1)
# vg0 = np.delete(vg0, np.where(vg0 == 0.0))
# plt.hist(vg0, bins = 100)
# plt.show()

f.close()
