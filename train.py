# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_da_model
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np
import torch
import torchvision.utils as vutils
from torch.autograd import Variable


opt = TrainOptions().parse()

if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(dataset) * opt.batchSize
print('#training images = %d' % dataset_size)

path = os.path.join(opt.checkpoints_dir, opt.name, 'model.txt')
visualizer = Visualizer(opt)

iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    visualizer.print_save('Resuming from epoch %d at iteration %d' % (start_epoch - 1, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

# opt.which_epoch=start_epoch-1

model = create_da_model(opt)
#fd = open(path, 'w')
#fd.write(str(model.module.netG))
#fd.write(str(model.module.netD))
#fd.close()

total_steps = (start_epoch - 1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

for epoch in range(61, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        ############## Forward Pass ######################
        # newpath = '/home/dsplxp/code/code/dehaze/Bringing-Old-Photos-Back-to-Life-master/Global/savehaze/'
        # img.save(newpath + 'haze.png')

        losses, generated, fake_image_clean, fake_image_real = model(Variable(data['label']), Variable(data['inst']), Variable(data['image']), Variable(data['feat']), Variable(data['real']), infer=save_fake)
        # sum per device losses
        losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar
        '''if epoch > 120:
                                    loss_G = loss_dict['loss_step2']
                                    model.module.optimizer_mapping.zero_grad()
                                    loss_G.backward()
                                    model.module.optimizer_mapping.step()'''


        if epoch> 60:
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_D_real = (loss_dict['D_fake_real'] + loss_dict['D_real_real']) * 0.5
            loss_featD = (loss_dict['featD_fake'] + loss_dict['featD_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict['G_GAN_real'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_GAN_Feat_real', 0) + loss_dict.get('G_VGG', 0) + loss_dict.get('G_VGG_real', 0) + loss_dict['G_KL'] + loss_dict['G_KL_real'] + loss_dict['G_featD'] + loss_dict['G_GAN_dec'] + loss_dict['G_GAN_Feat_dec'] + loss_dict['G_VGG_dec'] + loss_dict['G_GAN_Feat_dec'] +loss_dict['smooth_l1_loss'] + loss_dict['Diff'] + loss_dict['loss_gt'] + loss_dict['loss_G_featD_real']+loss_dict['loss_G_GAN_gt2real']
            loss_D_dec = loss_dict['D_real_dec'] + loss_dict['D_fake_dec']
            loss_D_gt2real = loss_dict['loss_step2_gt2real']
            loss_D_real2gt = loss_dict['loss_step2_gtandreal']
            model.module.optimizer_G.zero_grad()
            model.module.optimizer_PSG.zero_grad()
            model.module.optimizer_PRG.zero_grad()
            loss_G.backward()
            model.module.optimizer_G.step()
            model.module.optimizer_PSG.step()
            model.module.optimizer_PRG.step()

            # update discriminator weights
            model.module.optimizer_D.zero_grad()
            loss_D.backward()
            model.module.optimizer_D.step()

            model.module.optimizer_D_real.zero_grad()
            loss_D_real.backward()
            model.module.optimizer_D_real.step()

            model.module.optimizer_featD.zero_grad()
            loss_featD.backward()
            model.module.optimizer_featD.step()

            model.module.optimizer_D_dec.zero_grad()
            loss_D_dec.backward()
            model.module.optimizer_D_dec.step()

            model.module.optimizer_D_gt2real.zero_grad()
            loss_D_gt2real.backward()
            model.module.optimizer_D_gt2real.step()

            model.module.optimizer_D_real2gt.zero_grad()
            loss_D_real2gt.backward()
            model.module.optimizer_D_real2gt.step()


        else :
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_D_real = (loss_dict['D_fake_real'] + loss_dict['D_real_real']) * 0.5
            loss_featD = (loss_dict['featD_fake'] + loss_dict['featD_real']) * 0.5

            loss_G = loss_dict['G_GAN'] + loss_dict['G_GAN_real'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get(
                'G_GAN_Feat_real', 0) + loss_dict.get('G_VGG', 0) + loss_dict.get('G_VGG_real', 0) + loss_dict['G_KL'] + loss_dict['G_KL_real'] + loss_dict['G_featD'] + loss_dict['G_GAN_dec'] + loss_dict[
                         'G_GAN_Feat_dec'] + loss_dict['G_VGG_dec'] + loss_dict['G_GAN_Feat_dec'] + loss_dict[
                         'smooth_l1_loss'] + loss_dict['Diff']

            loss_D_dec = loss_dict['D_real_dec'] + loss_dict['D_fake_dec']
            model.module.optimizer_G.zero_grad()
            model.module.optimizer_PSG.zero_grad()
            model.module.optimizer_PRG.zero_grad()
            loss_G.backward()
            model.module.optimizer_G.step()
            model.module.optimizer_PSG.step()
            model.module.optimizer_PRG.step()

            # update discriminator weights
            model.module.optimizer_D.zero_grad()
            loss_D.backward()
            model.module.optimizer_D.step()

            model.module.optimizer_D_real.zero_grad()
            loss_D_real.backward()
            model.module.optimizer_D_real.step()

            model.module.optimizer_featD.zero_grad()
            loss_featD.backward()
            model.module.optimizer_featD.step()

            model.module.optimizer_D_dec.zero_grad()
            loss_D_dec.backward()
            model.module.optimizer_D_dec.step()




        # call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t, model.module.old_lr)
            visualizer.plot_current_errors(errors, total_steps)

        ### display output images
        if save_fake:

            if not os.path.exists(opt.outputs_dir + opt.name):
                os.makedirs(opt.outputs_dir + opt.name)
            imgs_num = data['label'].shape[0]
            imgs = torch.cat((data['label'], generated.data.cpu(), data['image'],fake_image_clean.data.cpu(), data['real'], fake_image_real.data.cpu()), 0)

            imgs = (imgs + 1.) / 2.0

            try:
                image_grid = vutils.save_image(imgs, opt.outputs_dir + opt.name + '/' + str(epoch) + '_' + str(
                    total_steps) + '.png',
                                               nrow=imgs_num, padding=0, normalize=True)
            except OSError as err:
                print(err)


        if epoch_iter >= dataset_size:
            break

    # end of epoch
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()

