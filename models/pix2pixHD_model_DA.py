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
from models.pytorch_spl_loss import GPLoss,CPLoss

class Mapping_Model(nn.Module):
    def __init__(self, nc, mc=64, n_blocks=3, norm="instance", padding_type="reflect", opt=None):
        super(Mapping_Model, self).__init__()

        norm_layer = networks.get_norm_layer(norm_type=norm)
        activation = nn.ReLU(True)
        model = []
        tmp_nc = 64
        n_up = 4

        for i in range(n_up):
            ic = min(tmp_nc * (2 ** i), mc)
            oc = min(tmp_nc * (2 ** (i + 1)), mc)
            model += [nn.Conv2d(ic, oc, 3, 1, 1), norm_layer(oc), activation]
        for i in range(n_blocks):
            model += [
                networks.ResnetBlock(
                    mc,
                    padding_type=padding_type,
                    activation=activation,
                    norm_layer=norm_layer,
                    opt=opt,
                    dilation=opt.mapping_net_dilation,
                )
            ]

        for i in range(n_up - 1):
            ic = min(64 * (2 ** (4 - i)), mc)
            oc = min(64 * (2 ** (3 - i)), mc)
            model += [nn.Conv2d(ic, oc, 3, 1, 1), norm_layer(oc), activation]
        model += [nn.Conv2d(tmp_nc * 2, tmp_nc, 3, 1, 1)]
        if opt.feat_dim > 0 and opt.feat_dim < 64:
            model += [norm_layer(tmp_nc), activation, nn.Conv2d(tmp_nc, opt.feat_dim, 1, 1)]
        # model += [nn.Conv2d(64, 1, 1, 1, 0)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True)

        def loss_filter(g_gan, g_gan_real, g_gan_feat, g_gan_feat_real, g_vgg, g_vgg_real, g_kl, g_kl_real, d_real, d_real_real, d_fake, d_fake_real, g_featd, featd_real, featd_fake, diff, g_gan_dec, g_gan_feat_dec, g_vgg_dec, d_real_dec, d_fake_dec, smooth_l1_loss, l_gt, step2_gtandreal, step2_gt2real, featD_real, GAN_gt2real):
            return [l for (l, f) in zip((g_gan, g_gan_real, g_gan_feat, g_gan_feat_real, g_vgg, g_vgg_real, g_kl, g_kl_real, d_real, d_real_real, d_fake,  d_fake_real, g_featd, featd_real, featd_fake, diff,
                                         g_gan_dec, g_gan_feat_dec, g_vgg_dec, d_real_dec, d_fake_dec, smooth_l1_loss, l_gt, step2_gtandreal, step2_gt2real, featD_real, GAN_gt2real), flags) if f]

        return loss_filter

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain:  # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat  ## Clearly it is false
        self.gen_features = self.use_features and not self.opt.load_features  ## it is also false
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc  ## Just is the origin input channel #

        ##### define networks
        # Generator network
        netG_input_nc = input_nc
        if not opt.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, opt.k_size,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids, opt=opt)

        # haze private Generator network
        self.Privatesys_netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, opt.k_size,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids, opt=opt)

        self.Privatereal_netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, opt.k_size,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids, opt=opt)

        '''self.mapping_net = Mapping_Model(
                                    min(opt.ngf * 2 ** opt.n_downsample_global, opt.mc),
                                    opt.map_mc,
                                    n_blocks=opt.mapping_n_block,
                                    opt=opt,
                                )
                        
                                if len(self.gpu_ids) > 0:
                                    assert (torch.cuda.is_available())
                                    self.mapping_net.cuda(self.gpu_ids[0])
                                self.mapping_net.apply(networks.weights_init)'''


        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            netD_input_nc_de = opt.ngf * 2 if opt.feat_gan else input_nc + opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt,opt.norm, use_sigmoid,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

            self.netD_real = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt, opt.norm, use_sigmoid,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

            self.netD_gt2real = networks.define_D(3, opt.ndf, opt.n_layers_D, opt, opt.norm, use_sigmoid,
                                          1, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

            self.netD_real2gt = networks.define_D(64, opt.ndf, opt.n_layers_D, opt, opt.norm, use_sigmoid,
                                          1, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)




            self.feat_D=networks.define_D(64, opt.ndf, opt.n_layers_D, opt, opt.norm, use_sigmoid,
                                          1, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

            self.netD_dec = networks.define_D(netD_input_nc_de, opt.ndf, opt.n_layers_D, opt, opt.norm, use_sigmoid,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)




        if self.opt.verbose:
            print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            self.load_network(self.Privatesys_netG, 'PSG', opt.which_epoch, pretrained_path)
            self.load_network(self.Privatereal_netG, 'PRG', opt.which_epoch, pretrained_path)
            #self.load_network(self.mapping_net, "mapping_net", opt.which_epoch, pretrained_path)
            print("---------- G Networks reloaded -------------")
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
                self.load_network(self.netD_real2gt, 'D_real2gt', opt.which_epoch, pretrained_path)
                self.load_network(self.netD_real, 'D_real', opt.which_epoch, pretrained_path)
                self.load_network(self.feat_D, 'feat_D', opt.which_epoch, pretrained_path)
                self.load_network(self.netD_dec, 'D_dec', opt.which_epoch, pretrained_path)
                self.load_network(self.netD_gt2real, 'D_gt2real', opt.which_epoch, pretrained_path)

                print("---------- D Networks reloaded -------------")


                print("---------- G Private Networks reloaded -------------")





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
            self.loss_diff1 = networks.DiffLoss()
            self.GPL = GPLoss()
            self.CPL = CPLoss(rgb=True,yuv=True,yuvgrad=True)
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss_torch(self.gpu_ids)
            if self.opt.image_L1:
                self.criterionImage=torch.nn.L1Loss()
            else:
                self.criterionImage = torch.nn.SmoothL1Loss()

            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN', 'G_GAN_real', 'G_GAN_Feat', 'G_GAN_Feat_real', 'G_VGG', 'G_VGG_real', 'G_KL', 'G_KL_real', 'D_real', 'D_real_real', 'D_fake', 'D_fake_real', 'G_featD', 'featD_real','featD_fake','Diff', 'G_GAN_dec','G_GAN_Feat_dec','G_VGG_dec','D_real_dec','D_fake_dec','smooth_l1_loss','loss_gt','loss_step2_gtandreal', 'loss_step2_gt2real', 'loss_G_featD_real', 'loss_G_GAN_gt2real')

            # initialize optimizers
            # optimizer G
            params = list(self.netG.parameters())

            if self.gen_features:
                params += list(self.netE.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer private sys G
            params = list(self.Privatesys_netG.parameters())
            self.optimizer_PSG = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            #params = list(self.mapping_net.parameters())
            #self.optimizer_mapping = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer private real G
            params = list(self.Privatereal_netG.parameters())
            self.optimizer_PRG = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            params = list(self.netD_real2gt.parameters())
            self.optimizer_D_real2gt = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))


            params = list(self.netD_gt2real.parameters())
            self.optimizer_D_gt2real = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))


            params = list(self.netD_real.parameters())
            self.optimizer_D_real = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            params = list(self.feat_D.parameters())
            self.optimizer_featD = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            params = list(self.netD_dec.parameters())
            self.optimizer_D_dec = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            print("---------- Optimizers initialized -------------")

            if opt.continue_train:
                self.load_optimizer(self.optimizer_D, 'D', opt.which_epoch)
                self.load_optimizer(self.optimizer_D_real, 'D_real', opt.which_epoch)
                self.load_optimizer(self.optimizer_G, "G", opt.which_epoch)
                self.load_optimizer(self.optimizer_PSG, "PSG", opt.which_epoch)
                self.load_optimizer(self.optimizer_PRG, "PRG", opt.which_epoch)
                self.load_optimizer(self.optimizer_featD,'featD',opt.which_epoch)
                self.load_optimizer(self.optimizer_D_dec, 'D_dec', opt.which_epoch)
                self.load_optimizer(self.optimizer_D_real2gt, 'D_real2gt', opt.which_epoch)
                #self.load_optimizer(self.optimizer_mapping, 'mapping_net', opt.which_epoch)
                self.load_optimizer(self.optimizer_D_gt2real, 'D_gt2real', opt.which_epoch)

                for param_groups in self.optimizer_D.param_groups:
                    self.old_lr = param_groups['lr']

                print("---------- Optimizers reloaded -------------")
                print("---------- Current LR is %.8f -------------" % (self.old_lr))

            ## We also want to re-load the parameters of optimizer.

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, real_haze_map=None, infer=False):
        if self.opt.label_nc == 0:
            input_label = label_map.data.cuda()
            real_haze = real_haze_map.data.cuda()
        else:
            # create one-hot vector for label map
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)

            size_real = real_haze_map.size()
            oneHot_size_real = (size_real[0], self.opt.label_nc, size_real[2], size_real[3])
            real_haze = torch.cuda.FloatTensor(torch.Size(oneHot_size_real)).zero_()
            real_haze = real_haze.scatter_(1, real_haze_map.data.long().cuda(), 1.0)

            if self.opt.data_type == 16:
                input_label = input_label.half()
                real_haze = real_haze.half()

        # get edges from instance map
        if not self.opt.no_instance:
            inst_map = inst_map.data.cuda()
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)
            real_haze = torch.cat((real_haze, edge_map), dim=1)
        input_label = Variable(input_label, volatile=infer)
        real_haze = Variable(real_haze, volatile=infer)

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())

        # instance map for feature encoding
        if self.use_features:
            # get precomputed feature maps
            if self.opt.load_features:
                feat_map = Variable(feat_map.data.cuda())
            if self.opt.label_feat:
                inst_map = label_map.cuda()

        return input_label, inst_map, real_image, feat_map, real_haze

    def discriminate(self, input_label, test_image, use_pool=False):
        if input_label is None:
            input_concat = test_image.detach()
        else:
            input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def discriminate_real(self, input_label, test_image, use_pool=False):
        if input_label is None:
            input_concat = test_image.detach()
        else:
            input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD_real.forward(fake_query)
        else:
            return self.netD_real.forward(input_concat)

    def discriminate_gt2real(self, input):

        return self.netD_gt2real.forward(input.detach())


    def discriminate_real2gt(self, input):
        return self.netD_real2gt.forward(input.detach())

    def feat_discriminate(self,input):

        return self.feat_D.forward(input.detach())

    def discriminate_dec(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD_dec.forward(fake_query)
        else:
            return self.netD_dec.forward(input_concat)


    def forward(self, label, inst, image, feat, real_ha, infer=False):
        # Encode Inputs
        input_label, inst_map, real_image, feat_map, real_haze = self.encode_input(label, inst, image, feat, real_ha)

        # Fake Generation
        if self.use_features:
            if not self.opt.load_features:
                feat_map = self.netE.forward(real_image, inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)
        else:
            input_concat = input_label
            input_concat_real = real_haze

        hiddens = self.netG.forward(input_concat, 'enc')
        real_hiddens = self.netG.forward(input_concat_real, 'enc')

        hiddens_sys = self.Privatesys_netG.forward(input_concat, 'enc')
        hiddens_gt = self.Privatesys_netG.forward(real_image, 'enc')

        loss_gt = self.criterionFeat(hiddens_sys,hiddens_gt)


        hiddens_real = self.Privatereal_netG.forward(input_concat_real, 'enc')
        hiddens_real_gt = self.Privatereal_netG.forward(real_image, 'enc')

        #hiddens_real2sys_gt = self.mapping_net(hiddens_real_gt)
        #hiddens_real2sys_real = self.mapping_net(hiddens_real)

        #loss_real2sys = self.criterionFeat(hiddens_real2sys_gt,hiddens_gt)




        noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
        # This is a reduced VAE implementation where we assume the outputs are multivariate Gaussian distribution with mean = hiddens and std_dev = all ones.
        # We follow the the VAE of MUNIT (https://github.com/NVlabs/MUNIT/blob/master/networks.py)
        fake_image_real = self.Privatereal_netG.forward(hiddens_real + real_hiddens + noise, 'dec')   #xiugai--------------------------------------------------------

        fake_image_sys = self.netG.forward(hiddens_sys + hiddens + noise, 'dec')            #xiugai--------------------------------------------------------
        fake_real_gt = self.Privatereal_netG.forward(hiddens_real_gt + real_hiddens + noise, 'dec')

        fake_image_clean = self.Privatesys_netG.forward(hiddens_sys, 'dec')
        #fake_hiddens_real2sys_gt = self.Privatesys_netG.forward(hiddens_real2sys_gt, 'dec')

        #fake_image_real_clean = self.Privatesys_netG.forward(hiddens_real2sys_real, 'dec')



        # label_feat_map = self.mapping_net(hiddens)
        # fake_image_clean=self.Privatesys_netG.forward(label_feat_map + noise, 'dec')
        # label_feat_map_real = self.mapping_net(real_hiddens)
        # fake_image_real_clean = self.Privatesys_netG.forward(label_feat_map_real + noise, 'dec')


        ####################
        ##### GAN for the intermediate feature
        real_old_feat =real_hiddens
        syn_feat = hiddens

        private_sys_old_feat = hiddens_sys
        private_real_old_feat = hiddens_real

        pred_fake_feat=self.feat_discriminate(real_old_feat)              # here, the content of synthetic is true, real one is false.
        loss_featD_fake = self.criterionGAN(pred_fake_feat, False)
        pred_real_feat=self.feat_discriminate(syn_feat)
        loss_featD_real = self.criterionGAN(pred_real_feat, True)         # discriminate the content of real and synthetic
        pred_fake_feat_G=self.feat_D.forward(real_old_feat)        #trying to narrow the gap between the reconstructed real non-foggy picture and the real one
        loss_G_featD=self.criterionGAN(pred_fake_feat_G,True)

        pred_fake_feat_real=self.discriminate_real2gt(hiddens_real)              # here, the content of synthetic is true, real one is false.
        loss_featD_fake_real = self.criterionGAN(pred_fake_feat_real, False)
        pred_real_feat_real=self.discriminate_real2gt(hiddens_real_gt)
        loss_featD_real_real = self.criterionGAN(pred_real_feat_real, True)         # discriminate the content of real and synthetic
        pred_fake_feat_G_real=self.feat_D.forward(hiddens_real)        #trying to narrow the gap between the reconstructed real non-foggy picture and the real one
        loss_G_featD_real=self.criterionGAN(pred_fake_feat_G_real,True)

        pred_fake_pool_gt2real = self.discriminate_gt2real(fake_real_gt)
        loss_D_fake_gt2real = self.criterionGAN(pred_fake_pool_gt2real, False)
        pred_real_gt2real = self.discriminate_gt2real(real_haze)
        loss_D_real_gt2real = self.criterionGAN(pred_real_gt2real, True)
        pred_fake_gt2real = self.netD_gt2real.forward(fake_real_gt)
        loss_G_GAN_gt2real = self.criterionGAN(pred_fake_gt2real, True)



        ylong = int(self.opt.batchSize / len(self.opt.gpu_ids))
        y = torch.LongTensor([-1] * ylong)
        y = torch.autograd.Variable(y)
        y = y.cuda()
        #to enlarge the gap between encoded fog and encoded content as much as possible
        loss_diff_real = self.loss_diff(private_real_old_feat.view(private_real_old_feat.size(0), -1),real_old_feat.view(real_old_feat.size(0), -1),y)
        loss_diff_sys = self.loss_diff(private_sys_old_feat.view(private_sys_old_feat.size(0), -1), syn_feat.view(syn_feat.size(0), -1), y)
        loss_dif = 0.5* (loss_diff_real + loss_diff_sys)


        #####################################----------------------------------------------------------------------here, new codes!
        if self.opt.no_cgan:
            # Fake Detection and Loss
            # trying to discriminate the fake images and the real images (the reconstructed and the input ones)
            pred_fake_pool_sys = self.discriminate(input_label, fake_image_sys, use_pool=True)
            pred_fake_pool_real = self.discriminate_real(real_haze, fake_image_real, use_pool=True)
            #print('pred_fake_pool_sys:',pred_fake_pool_sys.shape)
            loss_D_fake = self.criterionGAN(pred_fake_pool_sys, False)
            loss_D_fake_real = self.criterionGAN(pred_fake_pool_real, False)
            # Real Detection and Loss
            pred_real_sys = self.discriminate(input_label, input_label)
            pred_real_real = self.discriminate_real(real_haze, real_haze)
            loss_D_real = self.criterionGAN(pred_real_sys, True)
            loss_D_real_real = self.criterionGAN(pred_real_real, True)
            # trying to fool the discriminator not to discriminate the fake and the real
            # GAN loss (Fake Passability Loss)
            pred_fake_sys = self.netD.forward(torch.cat((input_label, fake_image_sys), dim=1))
            pred_fake_real = self.netD_real.forward(torch.cat((real_haze, fake_image_real), dim=1))
            #print('pred_fake_sys:',pred_fake_sys.shape)
            loss_G_GAN = self.criterionGAN(pred_fake_sys, True)
            loss_G_GAN_real = self.criterionGAN(pred_fake_real, True)




            # Another unit of discriminate and anti-discriminate
            # Fake Detection and Loss
            pred_fake_pool_dec = self.discriminate_dec(input_label, fake_image_clean, use_pool=True)
            loss_D_fake_dec = self.criterionGAN(pred_fake_pool_dec, False)
            # Real Detection and Loss
            pred_real_dec = self.discriminate_dec(input_label, real_image)
            loss_D_real_dec = self.criterionGAN(pred_real_dec, True)
            # Fooling process of generator
            # GAN loss (Fake Passability Loss)
            pred_fake_dec = self.netD_dec.forward(torch.cat((input_label, fake_image_clean), dim=1))
            loss_G_GAN_dec = self.criterionGAN(pred_fake_dec, True)

        else:
            # Fake Detection and Loss
            pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)
            loss_D_fake = self.criterionGAN(pred_fake_pool, False)

            # Real Detection and Loss
            pred_real = self.discriminate(input_label, real_image)
            loss_D_real = self.criterionGAN(pred_real, True)

            # GAN loss (Fake Passability Loss)
            pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))

            loss_G_GAN = self.criterionGAN(pred_fake, True)


        loss_G_kl = torch.mean(torch.pow(hiddens, 2)) * self.opt.kl
        loss_G_kl_real = torch.mean(torch.pow(real_hiddens, 2)) * self.opt.kl

        # GAN feature matching loss
        loss_G_GAN_Feat_sys = 0
        loss_G_GAN_Feat_real = 0
        loss_G_GAN_Feat_dec = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake_sys[i]) - 1):
                    loss_G_GAN_Feat_sys += D_weights * feat_weights * \
                                       self.criterionFeat(pred_fake_sys[i][j],
                                                          pred_real_sys[i][j].detach()) * self.opt.lambda_feat

            for i in range(self.opt.num_D):
                for j in range(len(pred_fake_real[i]) - 1):
                    loss_G_GAN_Feat_real += D_weights * feat_weights * \
                                       self.criterionFeat(pred_fake_real[i][j],
                                                          pred_real_real[i][j].detach()) * self.opt.lambda_feat


            loss_G_GAN_Feat= loss_G_GAN_Feat_sys
            loss_G_GAN_Feat_real = loss_G_GAN_Feat_real


            for i in range(self.opt.num_D):
                for j in range(len(pred_fake_dec[i])-1):
                    tmp = self.criterionFeat(pred_fake_dec[i][j], pred_real_dec[i][j].detach()) * self.opt.lambda_feat
                    loss_G_GAN_Feat_dec += D_weights * feat_weights * tmp
        else:
            loss_G_GAN_Feat = torch.zeros(1).to(label.device)

        # VGG feature matching loss
        loss_G_VGG = 0
    #    loss_G_VGG_sys = 0
     #   loss_G_VGG_real = 0
        loss_G_VGG_dec=0

        if not self.opt.no_vgg_loss:

            loss_G_VGG_sys = self.criterionVGG(fake_image_sys, input_label) * self.opt.lambda_feat
            loss_G_VGG_real = self.criterionVGG(fake_image_real, real_haze) * self.opt.lambda_feat
            loss_G_VGG_dec = self.criterionVGG(fake_image_clean, real_image) * self.opt.lambda_feat

            loss_G_VGG = loss_G_VGG_sys
            loss_G_VGG_real = loss_G_VGG_real
        # Only return the fake_B image if necessary to save BW
            fake_image=fake_image_sys

        # color profile loss, a very strong constraint for the color and structure of in/out--------------------------------------------------------------------
        # gpl_value = GPL(generated,target)
        # cpl_value = CPL(generated,target)
        # spl_value = gpl_value + cpl_value

        smooth_l1_loss=0
        if self.opt.Smooth_L1:
            smooth_l1_loss=self.criterionImage(fake_image_clean,real_image)*self.opt.L1_weight
            #smooth_l1_loss2 = self.criterionImage(fake_hiddens_real2sys_gt, real_image) * self.opt.L1_weight
        #loss_step2 = smooth_l1_loss2 #+ loss_real2sys
        loss_step2_gtandreal = 0.5*(loss_featD_fake_real + loss_featD_real_real)
        loss_step2_gt2real = 0.5*(loss_D_fake_gt2real + loss_D_real_gt2real)

        return [self.loss_filter(loss_G_GAN, loss_G_GAN_real, loss_G_GAN_Feat, loss_G_GAN_Feat_real, loss_G_VGG, loss_G_VGG_real, loss_G_kl, loss_G_kl_real, loss_D_real, loss_D_real_real, loss_D_fake, loss_D_fake_real, loss_G_featD, loss_featD_real, loss_featD_fake, loss_dif, loss_G_GAN_dec,loss_G_GAN_Feat_dec,loss_G_VGG_dec,loss_D_real_dec,loss_D_fake_dec,smooth_l1_loss, loss_gt, loss_step2_gtandreal, loss_step2_gt2real, loss_G_featD_real, loss_G_GAN_gt2real),
                None if not infer else fake_image, fake_image_clean, fake_image_real]

    # def inference(self, label, inst, image=None, feat=None):
    #     # Encode Inputs
    #     image = Variable(image) if image is not None else None
    #     input_label, inst_map, real_image, _ = self.encode_input(Variable(label), Variable(inst), image, infer=True)
    #     print('555555555555555')
    #
    #     # Fake Generation
    #     if self.use_features:
    #         if self.opt.use_encoded_image:
    #             # encode the real image to get feature map
    #             feat_map = self.netE.forward(real_image, inst_map)
    #         else:
    #             # sample clusters from precomputed features
    #             feat_map = self.sample_features(inst_map)
    #         input_concat = torch.cat((input_label, feat_map), dim=1)
    #     else:
    #         input_concat = input_label
    #
    #     if torch.__version__.startswith('0.4'):
    #         with torch.no_grad():
    #             fake_image = self.netG.forward(input_concat)
    #     else:
    #         #fake_image = self.netG.forward(input_concat)
    #         real_map = self.netG.forward(input_concat,'enc')
    #         real_map_temp = self.mapping_net(real_map)
    #         fake_image = self.Privatesys_netG.forward(real_map_temp, 'dec')
    #     return fake_image

    def inference(self, label, inst):

        use_gpu = len(self.opt.gpu_ids) > 0
        if use_gpu:
            input_concat = label.data.cuda()
            inst_data = inst.cuda()
        else:
            input_concat = label.data
            inst_data = inst

        input_concat = Variable(input_concat, volatile=True)
        real_map = self.Privatesys_netG.forward(input_concat,'enc')
        fake_image = self.Privatesys_netG.forward(real_map, 'dec')
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

    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num + 1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i // 1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[num // 2, :]
            val = np.zeros((1, feat_num + 1))
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

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
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        self.save_network(self.netD_real, 'D_real', which_epoch, self.gpu_ids)
        self.save_network(self.feat_D,'featD',which_epoch,self.gpu_ids)
        self.save_network(self.Privatesys_netG, 'PSG', which_epoch, self.gpu_ids)
        self.save_network(self.Privatereal_netG, 'PRG', which_epoch, self.gpu_ids)
        self.save_network(self.netD_dec, 'D_dec', which_epoch, self.gpu_ids)

        self.save_network(self.netD_real2gt, 'D_real2gt', which_epoch, self.gpu_ids)
        #self.save_network(self.mapping_net, 'mapping_net', which_epoch, self.gpu_ids)
        self.save_network(self.netD_gt2real, 'D_gt2real', which_epoch, self.gpu_ids)




        self.save_optimizer(self.optimizer_G, "G", which_epoch)
        self.save_optimizer(self.optimizer_D, "D", which_epoch)
        self.save_optimizer(self.optimizer_D_real, "D_real", which_epoch)
        self.save_optimizer(self.optimizer_featD,'featD',which_epoch)
        self.save_optimizer(self.optimizer_PSG, "PSG", which_epoch)
        self.save_optimizer(self.optimizer_PRG, "PRG", which_epoch)
        self.save_optimizer(self.optimizer_D_dec, "D_dec", which_epoch)

        self.save_optimizer(self.optimizer_D_real2gt, "D_real2gt", which_epoch)
        #self.save_optimizer(self.optimizer_mapping, "mapping_net", which_epoch)
        self.save_optimizer(self.optimizer_D_gt2real, "D_gt2real", which_epoch)



        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):

        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D_real.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_featD.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr


class InferenceModel(Pix2PixHDModel):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)