import os
from glob import glob
import numpy as np
import torch.utils.data as data
from network import EyeNet
from netope import NetOpe
from dataloader import DataUtils, Gen

# if use cuda
use_cuda = True
# data directory
train_root = '/media/data_storage/ophthalmology/cornea_disease/'

####################### parameters ##############################################
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 5 folds
n_folds = 5
epochs = 40
batch_size = 16
size = 299
num_workers = 4
epoch_th = np.array([5, 20])
day = '20181115'
multi_task = True

if multi_task:
    model_name = 'eye_fore_multi_task'
else:
    model_name = 'eye_fore'

classes = ['Cataract', 'Normal_Surface', 'Cornea_Infectious',
           'Cornea_Non_Infectious', 'Cornea_Degeneration', 'Cornea_Neoplasm']

########### generate model name #####################################################
def get_save_model_path(name='patch', day='20180904', k=1, loop=0):
    return './checkpoint/' + name + '_' + day + '_k%d_loop%d.pth' % (k, loop)
    # return './checkpoint/' + name + '_' + day + '_k%d_epoch3.pth' % k

def main():
    # dict containing data of different catagories
    class_dict = {}
    for cls in classes:
        class_dict[cls] = sorted(glob(os.path.join(train_root + cls, '*')))

    # object of data gen
    data_utils = DataUtils(path_dict=class_dict, classes=classes, n_folds=5, seed=10)

    # train n_folds times for 5fold cross validation
    # for kth in range(n_folds):
    for kth in range(1):
        print('The %dth fold ...' % kth)
        model_path = get_save_model_path(name=model_name, day=day, k=kth, loop=0)
        print('model save path is: ', model_path)
        # init network
        net = EyeNet(classes=classes, pool_size1=1, pool_size2=8, branch_in=2048, multi_task=multi_task)
        netope = NetOpe(net, num_class=len(classes), classes=classes, size=size, model_path=model_path,
                        multi_task=multi_task, resume=False, always_save=False, use_cuda=use_cuda)

        # get data info of current fold
        cur_dic = data_utils.data_info['%dfold' % kth]
        cur_dic = data_utils.split_data_into_blocks(cur_dic)
        val_samples = cur_dic['X_val']
        val_labels = cur_dic['y_val']
        val_gen = Gen(val_samples, val_labels, classes=classes, multi_task=multi_task, size=size, type='val')
        val_loader = data.DataLoader(val_gen, batch_size=int(batch_size), shuffle=True, num_workers=num_workers,
                                     collate_fn=val_gen.collate_fn)
        val_num = int(val_gen.len / batch_size)
        for epoch in range(epochs):
            for counter in range(5):
                counter = epoch * 5 + counter
                train_samples, train_labels = data_utils.get_balanced_data(cur_dic, counter)
                train_gen = Gen(train_samples, train_labels, classes=classes, multi_task=multi_task, size=size,
                                type='train')
                train_loader = data.DataLoader(train_gen, batch_size=int(batch_size), shuffle=True,
                                               num_workers=num_workers,
                                               collate_fn=train_gen.collate_fn)
                train_num = int(train_gen.len / batch_size)
                netope.train(epoch=counter, train_loader=train_loader, num=train_num, epoch_th=epoch_th * 5)
            netope.val(epoch=counter, test_loader=val_loader, num=val_num, metrics='auc')


if __name__ == '__main__':
    main()
