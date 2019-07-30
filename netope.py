import os
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from focalloss import FocalLoss


######### net operators ########################################################
class NetOpe:
    def __init__(self, net, num_class, classes, size, model_path, multi_task=False, resume=False, always_save=False, use_cuda=True):
        self.always_save = always_save
        self.size = size
        self.classes = classes
        self.model_path = model_path
        self.resume = resume
        self.start_epoch = 0
        self.multi_task = multi_task
        self.best_loss = 1000000
        self.net = net
        self.use_cuda = use_cuda
        if not net is None and use_cuda:
            self.net.cuda()

        self.criterion = FocalLoss(class_num=num_class, alpha=None, gamma=2, size_average=True)
        if self.resume and self.net is not None:
            self.start_epoch, self.best_loss = self.resume_model(self.model_path)

    def _update_model_param(self, state_dict, prefix):
        # 新模型参数dict
        net_state_dict = self.net.state_dict()
        # 将不用的key剥离
        state_dict = {prefix + k: v for k, v in state_dict.items() if prefix + k in net_state_dict}
        # print('same param names: ', state_dict.keys())
        # 更新到新模型中
        net_state_dict.update(state_dict)
        # 新模型加载参数
        self.net.load_state_dict(net_state_dict)

    def resume_model(self, model_path):
        print('==> Resuming from checkpoint')
        checkpoint = torch.load(model_path)
        self.net.load_state_dict(checkpoint['net'])
        best_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch']
        print('start_epoch:', start_epoch)
        return start_epoch, best_loss

    # 动态调整学习率
    # dynamically adjust learning rate
    def adjust_optim(self, epoch, epoch_th=[5, 15]):
        # lr = 1e-4 if epoch < epoch_th else 1e-5
        if epoch < epoch_th[0]:
            lr = 1e-4
        elif epoch < epoch_th[1]:
            lr = 1e-5
        else:
            lr = 1e-6

        if epoch == epoch_th[0]:
            print('learning rate: 1e-5')
            self.start_epoch, self.best_loss = self.resume_model(self.model_path)
        elif epoch == epoch_th[1]:
            print('learning rate: 1e-6')
            self.start_epoch, self.best_loss = self.resume_model(self.model_path)

        optimizer_pt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()),
                                        lr=lr, weight_decay=0.0005)
        return optimizer_pt

    # chose the parameter to train
    def split_param(self, chosen_param):
        chosen_id = []
        for cp in chosen_param:
            chosen_id += list(map(id, cp))
        chosen_p = filter(lambda p: id(p) in chosen_id, self.net.parameters())
        unchosen_p = filter(lambda p: id(p) not in chosen_id, self.net.parameters())
        return chosen_p, unchosen_p

    # training function
    def train(self, epoch, train_loader, num, epoch_th):
        print('train')
        self.net.train()

        for param in self.net.parameters():
            param.requires_grad = True

        if epoch < epoch_th[0]:
            for param in self.net.features.parameters():
                param.requires_grad = False
        if epoch == epoch_th[0]:
            print('unfreeze all parameters')

        optimizer_ft = self.adjust_optim(epoch, epoch_th)
        ave_loss = self._train_val(epoch, train_loader, num, optimizer_ft=optimizer_ft, type='train')

    # validation function
    def val(self, epoch, test_loader, num, metrics='loss'):
        print('val')
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False
        if metrics == 'loss':
            ave_loss = self._train_val(epoch, test_loader, num, optimizer_ft=None, type='val')
            ave_loss = np.mean(ave_loss)
        elif metrics == 'auc':
            predict_dict, predict_label_dict = self._train_val(epoch, test_loader, num, optimizer_ft=None,
                                                               type='predict')
            best_sens, best_spec, best_thre, aucc = self.sens_spec_for_dict(predict_dict, predict_label_dict)
            ave_loss = 1 - np.mean(aucc)
            str_sens = ','.join(['%.3f' % m for m in best_sens])
            str_spes = ','.join(['%.3f' % m for m in best_spec])
            str_thre = ','.join(['%.3f' % m for m in best_thre])
            str_aucc = ','.join(['%.3f' % m for m in aucc])
            print('epoch%d |ave_auc: %.3f' % (epoch, 1 - ave_loss),
                  '|sens:', str_sens, '|spes:', str_spes, '|thre:', str_thre, '|aucc:', str_aucc, end='\r')

        update = False
        if ave_loss < self.best_loss:
            update = True

            print('saving...')
            state = {
                'net': self.net.state_dict(),
                'loss': ave_loss,
                'epoch': epoch
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            if self.always_save:
                torch.save(state, self.model_path.split('.pth')[0] + '_epoch%d.pth' % epoch)
            else:
                torch.save(state, self.model_path)
            self.best_loss = ave_loss

        else:
            print()
        return update

    def _train_val(self, epoch, data_loader, num, optimizer_ft=None, type='train'):
        ave_loss = []
        sens, spes, accs = {}, {}, {}
        ave_sens, ave_spes, ave_accs = {}, {}, {}

        if type == 'predict':
            predict_dict = {}
            predict_label_dict = {}
            predict_middle_vector = {}

        for batch_idx, (inputs, labels, paths) in enumerate(data_loader):
            # print('batch_idx', batch_idx)
            if self.use_cuda:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels)

            if type == 'train':
                optimizer_ft.zero_grad()

            if len(inputs.size()) == 5:
                bs, ncrops, c, h, w = inputs.size()
                preds_n = self.net(inputs.view(-1, c, h, w))
                preds = preds_n.view(bs, ncrops, -1).mean(1)
            else:
                preds = self.net(inputs)
                middle_v = self.net.pool(self.net.features(inputs))
                middle_v = middle_v.cpu().numpy().squeeze()
            # print(preds.shape)
            # exit()
            if self.multi_task:
                loss = 0
                for s, name in enumerate(self.classes):
                    pred = preds[s]
                    label = torch.argmax(labels[:, s, :], dim=1)
                    loss += self.criterion(pred, label.long())
                label = labels.data.cpu().numpy()
                label = label[:, :, 0]
            else:
                loss = self.criterion(preds, torch.argmax(labels, dim=1))
                label = labels.data.cpu().numpy()

            if self.multi_task:
                pred = np.zeros(labels.shape[:2])
                for s, name in enumerate(self.classes):
                    p = preds[s]
                    p = F.softmax(p, dim=1).data.cpu().numpy()[:, 0]
                    pred[:, s] = p
            else:
                pred = F.softmax(preds, dim=1).data.cpu().numpy()

            if type == 'predict':
                for s, p in enumerate(pred):
                    cur_name = paths[s]
                    predict_dict[cur_name] = p
                    predict_middle_vector[cur_name] = middle_v[s]
                    predict_label_dict[cur_name] = label[s]

                print('%d/%d' % (batch_idx, num), end='\r')
            else:
                if type == 'train':
                    loss.backward()
                    optimizer_ft.step()

                loss = loss.data.cpu().numpy()
                ave_loss.append(loss)
                ave_loss_ = np.mean(ave_loss)

                for s, cls in enumerate(self.classes):
                    if not cls in sens.keys():
                        sens[cls], spes[cls], accs[cls] = [0, 0], [0, 0], [0, 0]
                    for m, p in enumerate(pred):
                        if np.argmax(p) == s:
                            sens[cls][0] += label[m][s]
                            accs[cls][0] += label[m][s]
                        else:
                            spes[cls][0] += np.sum(label[m]) - label[m][s]
                            accs[cls][0] += np.sum(label[m]) - label[m][s]

                    sens[cls][1] = sens[cls][1] + np.sum(label[:, s])
                    spes[cls][1] = spes[cls][1] + len(label) - np.sum(label[:, s])
                    accs[cls][1] += len(label)

                    ave_sens[cls] = '%.3f' % (sens[cls][0] / (sens[cls][1] + 0.001))
                    ave_spes[cls] = '%.3f' % (spes[cls][0] / (spes[cls][1] + 0.001))
                    ave_accs[cls] = '%.3f' % (accs[cls][0] / (accs[cls][1] + 0.001))

                str_sens = ','.join([ave_sens[m] for m in ave_sens.keys()])
                str_spes = ','.join([ave_spes[m] for m in ave_spes.keys()])
                str_accs = ','.join([ave_accs[m] for m in ave_accs.keys()])
                print('epoch%d-%d/%d |t_loss: %.3f |avg_loss: %.3f' % (epoch, batch_idx, num, loss, ave_loss_),
                      '|sens:', str_sens, '|spes:', str_spes, '|accs:', str_accs, end='\r')

        # torch.cuda.empty_cache()
        if type == 'predict':
            return predict_dict, predict_label_dict, predict_middle_vector
        return ave_loss