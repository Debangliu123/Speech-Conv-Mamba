# Created on 2024-05-04
# Author: debang

import os
import time
import torch
from pit_criterion import cal_loss
import matplotlib.pyplot as plt


class Solver(object):

    def __init__(self, data, model, optimizer, args, scheduler):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']

        self.save_logfolder = args.save_logfolder
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        # Training config
        self.use_cuda = args.use_cuda
        self.epochs = args.epochs
        self.half_lr = args.half_lr
        self.half_lr1 = 1
        self.havfnum = 0
        self.havfnumlimt= 1
        self.val_no_impv1 = 0
        self.early_stop = args.early_stop
        self.max_norm = args.max_norm
        # save and load model
        self.save_folder = args.save_folder
        self.checkpoint = args.checkpoint
        self.continue_from = args.continue_from
        self.model_path = args.model_path
        # logging
        self.print_freq = args.print_freq
        # visualizing loss using visdom
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)
        self.visdom = args.visdom
        self.visdom_epoch = args.visdom_epoch
        self.visdom_id = args.visdom_id
        self.training_detailData_path = os.path.join(args.save_logfolder, 'training_detailData.txt')
        if self.visdom:
            from visdom import Visdom
            self.vis = Visdom(env=self.visdom_id)
            self.vis_opts = dict(title=self.visdom_id,
                                 ylabel='Loss', xlabel='Epoch',
                                 legend=['train loss', 'cv loss'])
            self.vis_window = None
            self.vis_epochs = torch.arange(1, self.epochs + 1)

        self._reset()

    def _reset(self):
        # Reset
        if self.continue_from:
            print('Loading checkpoint model %s' % self.continue_from)
            package = torch.load(self.continue_from)
            self.model.module.load_state_dict(package['state_dict'])
            self.optimizer.load_state_dict(package['optim_dict'])
            self.start_epoch = int(package.get('epoch', 1))
            self.tr_loss[:self.start_epoch] = package['tr_loss'][:self.start_epoch]
            self.cv_loss[:self.start_epoch] = package['cv_loss'][:self.start_epoch]
        else:
            self.start_epoch = 0
        # Create save folder
        os.makedirs(self.save_logfolder, exist_ok=True)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.halving = False
        self.val_no_impv = 0
        self.val_no_impv1 = 0

    def train(self):
        # Train model multi-epoches
        tr_avg_loss_list = []
        val_loss_list = []
        epoch_list = []
        for epoch in range(self.start_epoch, self.epochs):
            epoch_list.append(epoch + 1)
            # Train one epoch
            print("Training...")
            filename = open(self.training_detailData_path, 'a+')
            filename.write("Training...")
            filename.write('\n')
            filename.close()

            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch)
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(
                epoch + 1, time.time() - start, tr_avg_loss))
            print('-' * 85)
            filename = open(self.training_detailData_path, 'a+')
            filename.write('-' * 85)
            filename.write('\n')
            filename.write('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                           'Train Loss {2:.3f}'.format(
                epoch + 1, time.time() - start, tr_avg_loss))
            filename.write('\n')
            filename.write('-' * 85)
            filename.write('\n')
            filename.close()

            tr_avg_loss_list.append(tr_avg_loss)
            # Save model each epoch
            if self.checkpoint:
                file_path = os.path.join(
                    self.save_logfolder, 'epoch%d.pth.tar' % (epoch + 1))
                torch.save(self.model.module.serialize(self.model.module,
                                                       self.optimizer, epoch + 1,
                                                       tr_loss=self.tr_loss,
                                                       cv_loss=self.cv_loss),
                           file_path)
                print('Saving checkpoint model to %s' % file_path)

            # Cross validation
            print('Cross validation...')
            filename = open(self.training_detailData_path, 'a+')
            filename.write("Cross validation...")
            filename.write('\n')
            filename.close()
            self.model.eval()  # Turn off Batchnorm & Dropout
            val_loss = self._run_one_epoch(epoch, cross_valid=True)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.3f}'.format(
                epoch + 1, time.time() - start, val_loss))
            print('-' * 85)
            filename = open(self.training_detailData_path, 'a+')
            filename.write('-' * 85)
            filename.write('\n')
            filename.write('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                           'Valid Loss {2:.3f}'.format(
                epoch + 1, time.time() - start, val_loss))
            filename.write('\n')
            filename.write('-' * 85)
            filename.write('\n')
            filename.close()

            val_loss_list.append(val_loss)
            # Adjust learning rate (halving)
            # if self.havfnum >= self.havfnumlimt and self.val_no_impv1 == 0: break
            if self.half_lr:
                if val_loss >= self.prev_val_loss:
                    self.val_no_impv += 1
                    if self.val_no_impv >= 3:
                        self.halving = True
                    if self.val_no_impv >= 8 and self.early_stop:
                        print("No imporvement for 10 epochs, early stopping.")
                        break
                else:
                    self.val_no_impv = 0

            # Adjust learning rate (halving)
            ###########################################
            if self.half_lr1:
                if val_loss >= self.best_val_loss:
                    self.val_no_impv1 += 1
                    if self.val_no_impv1 == 7:
                        optim_state = self.optimizer.state_dict()
                        optim_state['param_groups'][0]['lr'] = \
                            optim_state['param_groups'][0]['lr'] / 2.0
                        self.optimizer.load_state_dict(optim_state)
                        print('Learning rate adjusted to: {lr:.6f}'.format(
                            lr=optim_state['param_groups'][0]['lr']))
                        # self.val_no_impv1 = 0
                        self.havfnum += 1
                    if self.val_no_impv1 >= 8 and self.early_stop:
                        print("No imporvement for 10 epochs, early stopping.")
                        break
                else:
                    self.val_no_impv1 = 0
            ###########################################
            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = \
                    optim_state['param_groups'][0]['lr'] / 2.0
                self.optimizer.load_state_dict(optim_state)
                print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))
                self.halving = False
            self.prev_val_loss = val_loss

            # Save the best model
            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                file_path = os.path.join(self.save_logfolder, self.model_path)
                torch.save(self.model.module.serialize(self.model.module,
                                                       self.optimizer, epoch + 1,
                                                       tr_loss=self.tr_loss,
                                                       cv_loss=self.cv_loss),
                           file_path)
                print("Find better validated model, saving to %s" % file_path)

            # visualizing loss using visdom
            if self.visdom:
                x_axis = self.vis_epochs[0:epoch + 1]
                y_axis = torch.stack(
                    (self.tr_loss[0:epoch + 1], self.cv_loss[0:epoch + 1]), dim=1)
                if self.vis_window is None:
                    self.vis_window = self.vis.line(
                        X=x_axis,
                        Y=y_axis,
                        opts=self.vis_opts,
                    )
                else:
                    self.vis.line(
                        X=x_axis.unsqueeze(0).expand(y_axis.size(
                            1), x_axis.size(0)).transpose(0, 1),  # Visdom fix
                        Y=y_axis,
                        win=self.vis_window,
                        update='replace',
                    )
            # self.scheduler.step()
            optim_state = self.optimizer.state_dict()
            print('Learning rate adjusted to: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))
            print('self.best_val_loss:%.4f' % self.best_val_loss)
            filename = open(self.training_detailData_path, 'a+')
            filename.write('Learning rate adjusted to: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))
            filename.write('\n')
            filename.write('self.best_val_loss:%.4f' % self.best_val_loss)
            filename.write('\n')
            filename.close()

            plt.cla()
            plt.title('training loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.plot(epoch_list, val_loss_list, label='cv_avg_loss', color='#00CED1', linewidth=1)
            plt.plot(epoch_list, tr_avg_loss_list, label='tr_avg_loss', color='#DC143C', linewidth=1)
            plt.legend()
            plt.grid()
            training_loss_path = os.path.join(self.save_logfolder, 'training_loss.png')
            plt.savefig(training_loss_path, dip=400)
            train_data_path = os.path.join(self.save_logfolder, 'train_data.txt')
            filename = open(train_data_path, 'a+')
            filename.write(str('best_val_loss:'))
            filename.write(str(self.best_val_loss))
            filename.write('\n')
            filename.write(str(epoch + 1))
            filename.write('  ')
            filename.write(str(val_loss))
            filename.write('  ')
            filename.write('\n')
            filename.write(str(epoch + 1))
            filename.write('  ')
            filename.write(str(tr_avg_loss))
            filename.write('  ')
            filename.close()


    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0

        data_loader = self.tr_loader if not cross_valid else self.cv_loader
        # videoData = self.videoData

        # visualizing loss using visdom
        if self.visdom_epoch and not cross_valid:
            vis_opts_epoch = dict(title=self.visdom_id + " epoch " + str(epoch),
                                  ylabel='Loss', xlabel='Epoch')
            vis_window_epoch = None
            vis_iters = torch.arange(1, len(data_loader) + 1)
            vis_iters_loss = torch.Tensor(len(data_loader))

        for i, (data) in enumerate(data_loader):


            # GRID
            # padded_mixture, mixture_lengths, padded_source, _ = data
            # librimix
            padded_mixture, mixture_lengths, padded_source,_ = data



            if self.use_cuda:
                padded_mixture = padded_mixture.cuda()
                mixture_lengths = mixture_lengths.cuda()
                padded_source = padded_source.cuda()

            # TODO:videodata
            # start1 = time.time()
            estimate_source = self.model(padded_mixture)

            # print('estimate_source',estimate_source.shape)
            # print('padded_source',padded_source.shape)
            # end = time.time()
            # print("time",end-start)
            loss, max_snr, estimate_source, reorder_estimate_source = \
                cal_loss(padded_source, estimate_source, mixture_lengths)
            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.max_norm)
                self.optimizer.step()

            total_loss += loss.item()
            # start2 = time.time()
            # print('time:',start2-start1)
            if i % self.print_freq == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                      'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                    epoch + 1, i + 1, total_loss / (i + 1),
                    loss.item(), 1000 * (time.time() - start) / (i + 1)),
                    flush=True)
                filename = open(self.training_detailData_path, 'a+')
                filename.write('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                               'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                    epoch + 1, i + 1, total_loss / (i + 1),
                    loss.item(), 1000 * (time.time() - start) / (i + 1)))
                filename.write('\n')
                filename.close()

            # visualizing loss using visdom
            if self.visdom_epoch and not cross_valid:
                vis_iters_loss[i] = loss.item()
                if i % self.print_freq == 0:
                    x_axis = vis_iters[:i + 1]
                    y_axis = vis_iters_loss[:i + 1]
                    if vis_window_epoch is None:
                        vis_window_epoch = self.vis.line(X=x_axis, Y=y_axis,
                                                         opts=vis_opts_epoch)
                    else:
                        self.vis.line(X=x_axis, Y=y_axis, win=vis_window_epoch,
                                      update='replace')

        return total_loss / (i + 1)
