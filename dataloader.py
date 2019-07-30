import numpy as np
from sklearn.cross_validation import StratifiedKFold
import torch
from torchvision import transforms
from PIL import Image
import torch.utils.data as data


############ Data loader and kfold #############################################################
class DataUtils:
    def __init__(self, path_dict, classes, n_folds, seed=100):
        """
        data_info 是一个dict
            {'0fold':   {'class1': [paths], 'class2': [paths], ...
                         class_num: [num_class1, num_class2, ...]，
                         X_val: [paths], y_val: [y]},
             '1fold':   {'class1': [paths], 'class2': [paths], ...
                         class_num: [num_class1, num_class2, ...],
                         X_val: [paths], y_val: [y]},
             ...}
        patch_kfold_dict:
            { '0fold': {'train': {'class1': [patches], 'class2': [patches], ...},
                        'val':   {'class1': [patches], 'class2': [patches], ...}}
              '1fold': {'train': {'class1': [patches], 'class2': [patches], ...},
                        'val':   {'class1': [patches], 'class2': [patches], ...}}
              ...}

        :param path_dict:
                    { 'class1': [p1, p2,...],
                      'class2': [p1, p2,...],
                       ...}
        :param classes:
                    [  class1_name,   class2_name,  ... ]
        :param n_folds:
        :param seed:
        """
        self.patch_kfold_dict = {}
        self.classes = classes
        self.seed = seed
        self.data_info = {}
        self.n_folds = n_folds

        for cls in classes:
            print(cls, len(path_dict[cls]))

        ########################## stratifieldkfold ###################################
        paths = None
        y = None
        flag = False
        for s, c in enumerate(classes):
            cur_paths = path_dict[c]
            # cur_y = np.ones(len(cur_paths)) * s
            cur_y = np.array([c] * len(cur_paths))
            if not flag:
                paths = cur_paths
                y = cur_y
                flag = True
            else:
                paths = np.concatenate((paths, cur_paths))
                y = np.concatenate((y, cur_y))

        self.skf = StratifiedKFold(y, n_folds=n_folds, shuffle=True, random_state=seed)

        ######## Put each fold's paths into a new dict by category. ##################
        kth = 0
        for train_index, val_index in self.skf:
            print("TRAIN:", len(train_index), "TEST:", len(val_index))
            X_train, X_val = paths[train_index], paths[val_index]
            y_train, y_val = y[train_index], y[val_index]

            cur_kfold_dict = {'class_num': [], 'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val}
            for s, cur_y in enumerate(y_train):
                cur_class = cur_y
                if not cur_class in cur_kfold_dict.keys():
                    cur_kfold_dict[cur_class] = [X_train[s]]
                else:
                    cur_kfold_dict[cur_class].append(X_train[s])

            for c in self.classes:
                cur_kfold_dict['class_num'].append(len(cur_kfold_dict[c]))

            # Divide the data of each class into multiple points according to the
            # number of data of the smallest class.
            cur_kfold_dict = self.split_data_into_blocks(cur_kfold_dict)

            self.data_info['%dfold' % kth] = cur_kfold_dict

            kth += 1

    #################################################################################
    # Divide the data of each class into multiple points according to the
    # number of data of the smallest class.
    def split_data_into_blocks(self, cur_dict):
        min_num = np.min(cur_dict['class_num'])
        for s, c in enumerate(self.classes):
            cur_num = cur_dict['class_num'][s]
            accumulate_num = 0
            blocks = []
            while accumulate_num < cur_num:
                cur_block = cur_dict[c][accumulate_num: min(cur_num, accumulate_num + min_num)]
                accumulate_num += min_num
                if accumulate_num > cur_num:
                    cur_block.extend(cur_dict[c][: accumulate_num - cur_num])
                blocks.append(cur_block)
            if not c + '_blocks' in cur_dict.keys():
                cur_dict[c + '_blocks'] = blocks
            print('class %s has %d blocks' % (c, len(blocks)))
        return cur_dict

    ########### Tune the number of samples in each category to the same ##################
    def get_balanced_data(self, cur_kfold_dict, counter):
        selected_data = []
        selected_class = []
        # cur_kfold_dict = self.data_info['%dfold' % kth]
        for s, c in enumerate(self.classes):
            cur_blocks = cur_kfold_dict[c + '_blocks']
            cur_blocks_len = len(cur_blocks)
            residule = counter % cur_blocks_len
            selected_block = cur_blocks[residule]
            selected_data.extend(selected_block)
            selected_class.extend([c] * len(selected_block))
        selected_data = np.array(selected_data)
        selected_class = np.array(selected_class)
        return selected_data, selected_class


######### datagen class ########################################################
class Gen(data.Dataset):
    def __init__(self, samples, labels, classes, size=224, multi_task=False, type='train'):
        super(Gen, self).__init__()
        self.type = type
        self.samples = samples
        self.labels = labels
        self.classes = classes
        self.multi_task = multi_task
        if type == 'train':
            self.trans = self.trans_train(size)
        elif type == 'val':
            self.trans = self.trans_val(size)
        elif type == 'norm':
            self.trans = self.trans_norm(size)

        self.len = len(self.samples)

    def trans_train(self, size):
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomRotation(90),
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def trans_val(self, size):
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        cur_p = self.samples[idx]
        cur_label = self.labels[idx]
        cur_ind = self.classes.index(cur_label)
        if self.multi_task:
            label = [[0.0, 1.0]] * len(self.classes)
            label[cur_ind] = [1.0, 0.0]
        else:
            label = [0.0] * len(self.classes)
            label[cur_ind] = 1.0
        label = torch.Tensor(label)

        img = Image.open(cur_p).convert("RGB")
        img = self.trans(img)
        return img, label, cur_p

    def collate_fn(self, batch):
        imgs = [m[0] for m in batch]
        labels = [m[1] for m in batch]
        cur_ps = [m[2] for m in batch]
        return torch.stack(imgs), torch.stack(labels), cur_ps

    def __len__(self):
        return len(self.samples)

