# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import torch
import torch.nn as nn
from util.image_pool import ImagePool
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import DFF_DFM
from . import DFF_DFM_large
from . import DFF
from . import DFF_large

class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        #flags = (True, True, True, True, True,True, True, True, True, True, True,True, True, True, True, True, True, True, True, True, True, True, True, True, True)
        flags = (True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True)
        #        def loss_filter(sys_dehaze, D_cleanfeat_fake, D_cleanfeat_real, G_cleanfeat, D_noisefeat_fake, D_noisefeat_real, G_noisefeat, dif, sys_rec, D_sysrec_fake, D_sysrec_real, G_sysrec_GAN, G_GAN_Feat_sysrec, real_rec, D_recrealrec_fake, D_recrealrec_real, G_recrealrec_GAN, G_GAN_Feat_realrec, D_sysdehaze_fake, D_sysdehaze_real, G_sysdehaze_GAN, G_GAN_Feat_sysdehaze, G_VGG_sysdehaze, dark_channel, total_variation):
        #           return [l for (l, f) in zip((sys_dehaze, D_cleanfeat_fake, D_cleanfeat_real, G_cleanfeat, D_noisefeat_fake, D_noisefeat_real, G_noisefeat, dif, sys_rec, D_sysrec_fake, D_sysrec_real, G_sysrec_GAN, G_GAN_Feat_sysrec, real_rec, D_recrealrec_fake, D_recrealrec_real, G_recrealrec_GAN, G_GAN_Feat_realrec, D_sysdehaze_fake, D_sysdehaze_real, G_sysdehaze_GAN, G_GAN_Feat_sysdehaze, G_VGG_sysdehaze, dark_channel, total_variation), flags) if f]

       # def loss_filter(dif, sys_rec, real_rec, D_sysrec_fake, D_sysrec_real, G_sysrec_GAN, D_realrec_fake, D_realrec_real, G_realrec_GAN, D_sys2realrec_fake, D_sys2realrec_real, G_sys2realrec_GAN, D_real2sysrec_fake, D_real2sysrec_real, G_real2sysrec_GAN,G_GAN_Feat_sysrec, G_GAN_Feat_realrec, real_style, sys_style, real_content, sys_content, sys_dehaze, D_sysdehaze_fake, D_sysdehaze_real, G_sysdehaze_GAN, G_GAN_Feat_sysdehaze, G_VGG_sysdehaze, dark_channel, total_variation):
       #     return [l for (l, f) in zip((dif, sys_rec, real_rec, D_sysrec_fake, D_sysrec_real, G_sysrec_GAN, D_realrec_fake, D_realrec_real, G_realrec_GAN,D_sys2realrec_fake, D_sys2realrec_real, G_sys2realrec_GAN, D_real2sysrec_fake, D_real2sysrec_real, G_real2sysrec_GAN, G_GAN_Feat_sysrec, G_GAN_Feat_realrec, real_style, sys_style, real_content, sys_content, sys_dehaze, D_sysdehaze_fake, D_sysdehaze_real, G_sysdehaze_GAN, G_GAN_Feat_sysdehaze, G_VGG_sysdehaze, dark_channel, total_variation), flags) if f]

        def loss_filter(dif, sys_rec, real_rec, D_sysrec_fake, D_sysrec_real, G_sysrec_GAN, D_realrec_fake, D_realrec_real, G_realrec_GAN, D_sys2realrec_fake, D_sys2realrec_real, G_sys2realrec_GAN, D_real2sysrec_fake, D_real2sysrec_real, G_real2sysrec_GAN, dark_channel, total_variation,similiar,D_feat_fake,D_feat_real,G_feat,D_feat_gt_fake,D_feat_gt_real,G_feat_gt, G_GAN_Feat_gt, G_GAN_Feat_sysrec, G_GAN_Feat_realrec):
            return [l for (l, f) in zip((dif, sys_rec, real_rec, D_sysrec_fake, D_sysrec_real, G_sysrec_GAN, D_realrec_fake, D_realrec_real, G_realrec_GAN,D_sys2realrec_fake, D_sys2realrec_real, G_sys2realrec_GAN, D_real2sysrec_fake, D_real2sysrec_real, G_real2sysrec_GAN, dark_channel, total_variation, similiar,D_feat_fake,D_feat_real,G_feat,D_feat_gt_fake,D_feat_gt_real,G_feat_gt, G_GAN_Feat_gt, G_GAN_Feat_sysrec, G_GAN_Feat_realrec), flags) if f]


        return loss_filter

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain:  # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc  ## Just is the origin input channel #

        ##### define networks
        # Generator networknvidia-smi
        self.gtnetG = DFF.Net(output_nc = 3)

        self.imagenetG = DFF_large.Net(output_nc = 3)

        if len(self.gpu_ids) > 0:
            assert (torch.cuda.is_available())
            self.gtnetG.cuda(self.gpu_ids[0])
            self.imagenetG.cuda(self.gpu_ids[0])

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = 3
            self.feat_D = networks.define_featureD(128, n_layers=2, norm='batch', activation='PReLU',
													   init_type='xavier', gpu_ids=self.gpu_ids)

            self.feat_D_gt=networks.define_D(64, opt.ndf, opt.n_layers_D, opt, opt.norm, use_sigmoid,
                                          1, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

            self.sys_rec_netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt, opt.norm, use_sigmoid,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

            self.real_rec_netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt, opt.norm, use_sigmoid,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

            self.sys2real_rec_netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt, opt.norm, use_sigmoid,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

            self.real2sys_rec_netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt, opt.norm, use_sigmoid,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)


        if self.opt.verbose:
            print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.gtnetG, 'GG', opt.which_epoch, pretrained_path)
            self.load_network(self.imagenetG, 'IG', opt.which_epoch, pretrained_path)

            print("---------- G Networks reloaded -------------")
            if self.isTrain:
                self.load_network(self.feat_D, 'feat_D', opt.which_epoch, pretrained_path)
                self.load_network(self.feat_D_gt, 'feat_D_gt', opt.which_epoch, pretrained_path)
                self.load_network(self.sys_rec_netD, 'sys_rec_netD', opt.which_epoch, pretrained_path)
                self.load_network(self.real_rec_netD, 'real_rec_netD', opt.which_epoch, pretrained_path)
                self.load_network(self.sys2real_rec_netD, 'sys2real_rec_netD', opt.which_epoch, pretrained_path)
                self.load_network(self.real2sys_rec_netD, 'real2sys_rec_netD', opt.which_epoch, pretrained_path)


                # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:  ## The pool_size is 0!
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)

            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            self.loss_diff = torch.nn.CosineEmbeddingLoss(margin=0, size_average=True)
            self.TVLoss  = networks.L1_TVLoss_Charbonnier()
            self.loss_diff1 = networks.DiffLoss()
            self.loss_diff_3D = networks.Diff3D()
            #self.GPL = GPLoss()
            #self.CPL = CPLoss(rgb=True,yuv=True,yuvgrad=True)
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss_torch(self.gpu_ids)
                self.criterionSTYLE = networks.VGGLoss_style(self.gpu_ids)
                self.criterionCONTENT = networks.VGGLoss_content(self.gpu_ids)
            if self.opt.image_L1:
                self.criterionImage=torch.nn.L1Loss()
            else:
                self.criterionImage = torch.nn.SmoothL1Loss()

            # Names so we can breakout loss
            self.loss_names = self.loss_filter('dif', 'sys_rec', 'real_rec', 'D_sysrec_fake', 'D_sysrec_real', 'G_sysrec_GAN', 'D_realrec_fake', 'D_realrec_real', 'G_realrec_GAN', 'D_sys2realrec_fake', 'D_sys2realrec_real', 'G_sys2realrec_GAN', 'D_real2sysrec_fake', 'D_real2sysrec_real', 'G_real2sysrec_GAN', 'dark_channel', 'total_variation', 'similiar','D_feat_fake','D_feat_real','G_feat','D_feat_gt_fake','D_feat_gt_real','G_feat_gt', 'G_GAN_Feat_gt', 'G_GAN_Feat_sysrec', 'G_GAN_Feat_realrec')

            # initialize optimizers
            # optimizer G and decoder

            params = list(self.gtnetG.parameters())
            self.optimizer_GG = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            params = list(self.imagenetG.parameters())
            self.optimizer_IG = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))


            params = list(self.feat_D.parameters())
            self.optimizer_feat_D= torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            params = list(self.feat_D_gt.parameters())
            self.optimizer_feat_D_gt= torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            params = list(self.sys_rec_netD.parameters())
            self.optimizer_sys_rec_netD = torch.optim.Adam(params, lr=opt.lr/opt.lr_D, betas=(opt.beta1, 0.999))

            params = list(self.real_rec_netD.parameters())
            self.optimizer_real_rec_netD = torch.optim.Adam(params, lr=opt.lr/opt.lr_D, betas=(opt.beta1, 0.999))

            params = list(self.sys2real_rec_netD.parameters())
            self.optimizer_sys2real_rec_netD = torch.optim.Adam(params, lr=opt.lr/opt.lr_D, betas=(opt.beta1, 0.999))

            params = list(self.real2sys_rec_netD.parameters())
            self.optimizer_real2sys_rec_netD = torch.optim.Adam(params, lr=opt.lr/opt.lr_D, betas=(opt.beta1, 0.999))


            print("---------- Optimizers initialized -------------")

            if opt.continue_train:
                self.load_optimizer(self.optimizer_GG, "GG", opt.which_epoch)
                self.load_optimizer(self.optimizer_IG, "IG", opt.which_epoch)
                self.load_optimizer(self.optimizer_feat_D, 'feat_D', opt.which_epoch)
                self.load_optimizer(self.optimizer_feat_D_gt, 'feat_D_gt', opt.which_epoch)
                self.load_optimizer(self.optimizer_sys_rec_netD, 'sys_rec_netD', opt.which_epoch)
                self.load_optimizer(self.optimizer_real_rec_netD, 'real_rec_netD', opt.which_epoch)
                self.load_optimizer(self.optimizer_sys2real_rec_netD, 'sys2real_rec_netD', opt.which_epoch)
                self.load_optimizer(self.optimizer_real2sys_rec_netD, 'real2sys_rec_netD', opt.which_epoch)

                for param_groups in self.optimizer_IG.param_groups:
                    self.old_lr = param_groups['lr']

                print("---------- Optimizers reloaded -------------")
                print("---------- Current LR is %.8f -------------" % (self.old_lr))

            ## We also want to re-load the parameters of optimizer.

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, real_haze_map=None, density_syn_map=None, density_gt_map=None, density_real_map=None, infer=False):
        input_label = label_map.data.cuda()
        real_haze = real_haze_map.data.cuda()
        density_syn = density_syn_map.data.cuda()
        density_gt = density_gt_map.data.cuda()
        density_real= density_real_map.data.cuda()


        input_label = Variable(input_label, volatile=infer)
        real_haze = Variable(real_haze, volatile=infer)
        density_syn = Variable(density_syn, volatile=infer)
        density_gt = Variable(density_gt, volatile=infer)
        density_real = Variable(density_real, volatile=infer)

        real_image = Variable(real_image.data.cuda())


        return input_label, inst_map, real_image, feat_map, real_haze, density_syn, density_gt, density_real


    ######## feature discriminate ######
    def feat_discriminate(self,input):
        return self.feat_D.forward(input.detach())

    def feat_discriminate_gt(self,input):
        return self.feat_D_gt.forward(input.detach())

    ######## image discriminate ######

    def sys_rec_discriminate(self, input_label, test_image, use_pool=False):
        if input_label is None:
            input_concat = test_image.detach()
        else:
            input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.sys_rec_netD.forward(fake_query)
        else:
            return self.sys_rec_netD.forward(input_concat)

    def real_rec_discriminate(self, input_label, test_image, use_pool=False):
        if input_label is None:
            input_concat = test_image.detach()
        else:
            input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.real_rec_netD.forward(fake_query)
        else:
            return self.real_rec_netD.forward(input_concat)

    def sys2real_rec_discriminate(self, input_label, test_image, use_pool=False):
        if input_label is None:
            input_concat = test_image.detach()
        else:
            input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.sys2real_rec_netD.forward(fake_query)
        else:
            return self.sys2real_rec_netD.forward(input_concat)

    def real2sys_rec_discriminate(self, input_label, test_image, use_pool=False):
        if input_label is None:
            input_concat = test_image.detach()
        else:
            input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.real2sys_rec_netD.forward(fake_query)
        else:
            return self.real2sys_rec_netD.forward(input_concat)


    def DCLoss(self,img, res, patch_size):
        img = (img + 1.) / 2.0
        res = (res + 1.) / 2.0
        maxpool = torch.nn.MaxPool3d((3, patch_size, patch_size), stride=1, padding=(0, patch_size//2, patch_size//2))
        dc = maxpool(0-img)
        dc_shaped = dc.repeat(1, 3, 1, 1)
        res_shaped = torch.where(dc_shaped < -0.6, torch.zeros(res.shape).cuda(), res)
        target = torch.FloatTensor(res.shape).zero_().cuda()
        loss = self.criterionFeat(res_shaped, target)
        return loss


    def forward(self, label, inst, image, feat_map, real_haze, density_syn, density_gt, density_real, infer=False):
        # Encode Inputs
        input_sys, inst_map, gt, feat_map, input_real, density_sys, density_gt, density_real = self.encode_input(label, inst, image, feat_map, real_haze, density_syn, density_gt, density_real)
        ####   forward
        gt_clean = self.gtnetG.forward(gt, 'enc')
        sys_out = self.imagenetG.forward(input_sys, 'enc')
        real_out = self.imagenetG.forward(input_real,'enc')

        sys_clean = sys_out[1][:,:64,:,:]
        real_clean = real_out[1][:,:64,:,:]
        sys_noise = sys_out[1][:,64:,:,:]
        real_noise = real_out[1][:,64:,:,:]

        loss_similiar = 60 * self.criterionFeat(gt_clean, sys_clean)

        fake_image_sys = self.imagenetG.forward(sys_out[0],'dec')
        fake_image_real = self.imagenetG.forward(real_out[0], 'dec')
        fake_image_sys2real = self.imagenetG.forward(torch.cat((sys_clean, real_noise), dim=1), 'dec')
        fake_image_real2sys = self.imagenetG.forward(torch.cat((real_clean, sys_noise), dim=1), 'dec')

        fake_image_real_dehaze = self.gtnetG.forward(real_clean, 'dec')


        ## clean feature ##
        pred_fake_feat=self.feat_discriminate(real_out[0])                              # 暂时不拉近 real_clean 和 sys_clean
        loss_D_feat_fake = self.criterionGAN(pred_fake_feat, False)
        pred_real_feat=self.feat_discriminate(sys_out[0])
        loss_D_feat_real = self.criterionGAN(pred_real_feat, True)
        pred_fake_feat_G=self.feat_D.forward(real_out[0])
        loss_G_feat=self.criterionGAN(pred_fake_feat_G,True)


        pred_fake_feat_gt=self.feat_discriminate_gt(sys_clean)                              # 暂时不拉近 real_clean 和 sys_clean
        loss_D_feat_gt_fake = self.criterionGAN(pred_fake_feat_gt, False)
        pred_real_feat_gt=self.feat_discriminate_gt(gt_clean)
        loss_D_feat_gt_real = self.criterionGAN(pred_real_feat_gt, True)
        pred_fake_feat_gt_G=self.feat_D_gt.forward(sys_clean)
        loss_G_feat_gt=self.criterionGAN(pred_fake_feat_gt_G,True)




        loss_G_GAN_Feat_gt = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / 1
            for i in range(1):
                for j in range(len(pred_fake_feat_gt_G[i]) - 1):
                    loss_G_GAN_Feat_gt += D_weights * feat_weights * self.criterionFeat(pred_fake_feat_gt_G[i][j],
                                                                                            pred_real_feat_gt[i][
                                                                                                j].detach()) * self.opt.lambda_feat
        ############### distangled loss ###############
        ylong = int(self.opt.batchSize / len(self.opt.gpu_ids))
        y = torch.LongTensor([-1] * ylong)
        y = torch.autograd.Variable(y)
        y = y.cuda()
        #to enlarge the gap between encoded fog and encoded content as much as possible
        loss_diff_real = self.loss_diff(real_noise.view(real_noise.size(0), -1),real_clean.view(real_clean.size(0), -1),y)
        loss_diff_sys = self.loss_diff(sys_noise.view(sys_noise.size(0), -1), sys_clean.view(sys_clean.size(0), -1), y)
        loss_dif = 60.0* (loss_diff_real + loss_diff_sys)

        #loss_diff_real = self.loss_diff1(real_noise, real_clean)
        #loss_diff_sys = self.loss_diff1(sys_noise, sys_clean)
        #loss_dif = 5.0 * (loss_diff_real + loss_diff_sys)

        #loss_dif = 60*(self.loss_diff_3D(real_noise, real_clean) + self.loss_diff_3D(sys_noise, sys_clean))
        ##############      Loss for reconstruction      ####################
        ############### self reconstruction ###############
        ## vgg loss ##
        sys_rec_loss = self.criterionVGG(fake_image_sys, input_sys) * self.opt.lambda_feat
        real_rec_loss = self.criterionVGG(fake_image_real, input_real) * self.opt.lambda_feat

        ## GAN loss ##
        pred_fake_pool_sysrec = self.sys_rec_discriminate(None, fake_image_sys, use_pool=True)
        loss_D_sysrec_fake = self.criterionGAN(pred_fake_pool_sysrec, False)
        pred_real_sysrec = self.sys_rec_discriminate(None, input_sys)
        loss_D_sysrec_real = self.criterionGAN(pred_real_sysrec, True)
        pred_fake_sysrec = self.sys_rec_netD.forward(fake_image_sys)
        loss_G_sysrec_GAN = self.criterionGAN(pred_fake_sysrec, True)

        pred_fake_pool_realrec = self.real_rec_discriminate(None, fake_image_real, use_pool=True)
        loss_D_realrec_fake = self.criterionGAN(pred_fake_pool_realrec, False)
        pred_real_realrec = self.real_rec_discriminate(None, input_real)
        loss_D_realrec_real = self.criterionGAN(pred_real_realrec, True)
        pred_fake_realrec = self.real_rec_netD.forward(fake_image_real)
        loss_G_realrec_GAN = self.criterionGAN(pred_fake_realrec, True)

        pred_fake_pool_sys2realrec = self.sys2real_rec_discriminate(None, fake_image_sys2real, use_pool=True)
        loss_D_sys2realrec_fake = self.criterionGAN(pred_fake_pool_sys2realrec, False)
        pred_real_sys2realrec = self.sys2real_rec_discriminate(None, input_real)
        loss_D_sys2realrec_real = self.criterionGAN(pred_real_sys2realrec, True)
        pred_fake_sys2realrec = self.sys2real_rec_netD.forward(fake_image_sys2real)
        loss_G_sys2realrec_GAN = self.criterionGAN(pred_fake_sys2realrec, True)

        pred_fake_pool_real2sysrec = self.real2sys_rec_discriminate(None, fake_image_real2sys, use_pool=True)
        loss_D_real2sysrec_fake = self.criterionGAN(pred_fake_pool_real2sysrec, False)
        pred_real_real2sysrec = self.real2sys_rec_discriminate(None, input_sys)
        loss_D_real2sysrec_real = self.criterionGAN(pred_real_real2sysrec, True)
        pred_fake_real2sysrec = self.real2sys_rec_netD.forward(fake_image_real2sys)
        loss_G_real2sysrec_GAN = self.criterionGAN(pred_fake_real2sysrec, True)

        # GAN feature matching loss
        loss_G_GAN_Feat_sysrec = 0
        loss_G_GAN_Feat_realrec = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake_sysrec[i]) - 1):
                    loss_G_GAN_Feat_sysrec += D_weights * feat_weights * self.criterionFeat(pred_fake_sysrec[i][j],
                                                                                            pred_real_sysrec[i][
                                                                                                j].detach()) * self.opt.lambda_feat
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake_realrec[i]) - 1):
                    loss_G_GAN_Feat_realrec += D_weights * feat_weights * self.criterionFeat(pred_fake_realrec[i][j], pred_real_realrec[i][j].detach()) * self.opt.lambda_feat

        ##############      Loss for dehaze      ############################
        ############### real dehaze ###############
        loss_dark_channel = self.DCLoss(input_real, fake_image_real_dehaze, 3) * self.opt.lambda_DC
        loss_total_variation = self.TVLoss(fake_image_real_dehaze) * self.opt.lambda_TV

        return [self.loss_filter(loss_dif, sys_rec_loss, real_rec_loss, loss_D_sysrec_fake, loss_D_sysrec_real, loss_G_sysrec_GAN, loss_D_realrec_fake, loss_D_realrec_real, loss_G_realrec_GAN, loss_D_sys2realrec_fake, loss_D_sys2realrec_real, loss_G_sys2realrec_GAN, loss_D_real2sysrec_fake, loss_D_real2sysrec_real, loss_G_real2sysrec_GAN, loss_dark_channel, loss_total_variation, loss_similiar,loss_D_feat_fake,loss_D_feat_real,loss_G_feat,loss_D_feat_gt_fake,loss_D_feat_gt_real,loss_G_feat_gt, loss_G_GAN_Feat_gt, loss_G_GAN_Feat_sysrec, loss_G_GAN_Feat_realrec),
                None if not infer else fake_image_sys, fake_image_real, fake_image_real_dehaze, fake_image_sys2real, fake_image_real2sys]

    def inference(self, input, input_density):

        use_gpu = len(self.opt.gpu_ids) > 0
        if use_gpu:
            input = input.data.cuda()
        else:
            input = input.data

        input = Variable(input, volatile=True)
        real_map = self.imagenetG.forward(input,'enc')
        real_clean = real_map[1][:,:64,:,:]
        fake_image = self.gtnetG.forward(real_clean, 'dec')
        return fake_image

    def inference_cross(self, input):
        use_gpu = len(self.opt.gpu_ids) > 0
        if use_gpu:
            input = input.data.cuda()
        else:
            input = input.data

        input = Variable(input, volatile=True)
        real_map = self.imagenetG.forward(input,'enc')
        fake_image = self.imagenetG.forward(real_map[0],'dec')
        return fake_image

    def sample_features(self, inst):
        # read precomputed feature clusters
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)
        features_clustered = np.load(cluster_path, encoding='latin1').item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):
            label = i if i < 1000 else i // 1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0])

                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):
                    feat_map[idx[:, 0], idx[:, 1] + k, idx[:, 2], idx[:, 3]] = feat[cluster_idx, k]
        if self.opt.data_type == 16:
            feat_map = feat_map.half()
        return feat_map

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        if self.opt.data_type == 16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):
        self.save_network(self.gtnetG, 'GG', which_epoch, self.gpu_ids)
        self.save_network(self.imagenetG, 'IG', which_epoch, self.gpu_ids)

        self.save_network(self.feat_D, 'feat_D', which_epoch, self.gpu_ids)
        self.save_network(self.feat_D_gt, 'feat_D_gt', which_epoch, self.gpu_ids)

        self.save_network(self.sys_rec_netD, 'sys_rec_netD', which_epoch, self.gpu_ids)
        self.save_network(self.real_rec_netD, 'real_rec_netD', which_epoch, self.gpu_ids)
        self.save_network(self.sys2real_rec_netD, 'sys2real_rec_netD', which_epoch, self.gpu_ids)
        self.save_network(self.real2sys_rec_netD, 'real2sys_rec_netD', which_epoch, self.gpu_ids)

        self.save_optimizer(self.optimizer_GG, "GG", which_epoch)
        self.save_optimizer(self.optimizer_IG, "IG", which_epoch)

        self.save_optimizer(self.optimizer_feat_D, 'feat_D', which_epoch)
        self.save_optimizer(self.optimizer_feat_D_gt, 'feat_D_gt', which_epoch)
        self.save_optimizer(self.optimizer_sys_rec_netD, "sys_rec_netD", which_epoch)
        self.save_optimizer(self.optimizer_real_rec_netD, "real_rec_netD", which_epoch)
        self.save_optimizer(self.optimizer_sys2real_rec_netD, "sys2real_rec_netD", which_epoch)
        self.save_optimizer(self.optimizer_real2sys_rec_netD, "real2sys_rec_netD", which_epoch)



    def update_fixed_params(self):
        params = list(self.imagenetG.parameters())
        self.optimizer_IG = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_GG.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_IG.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_feat_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_feat_D_gt.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_sys_rec_netD.param_groups:
            param_group['lr'] = lr/self.opt.lr_D
        for param_group in self.optimizer_real_rec_netD.param_groups:
            param_group['lr'] = lr/self.opt.lr_D
        for param_group in self.optimizer_sys2real_rec_netD.param_groups:
            param_group['lr'] = lr/self.opt.lr_D
        for param_group in self.optimizer_real2sys_rec_netD.param_groups:
            param_group['lr'] = lr/self.opt.lr_D


        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr


class InferenceModel(Pix2PixHDModel):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)