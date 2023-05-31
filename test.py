# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from models.models import create_model
from models.pix2pixHD_model_DA import Pix2PixHDModel
import util.util as util
import PIL.Image as img
from PIL import Image
import pytorch_ssim
import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms
import numpy as np
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
from skimage.metrics import structural_similarity
import cv2

def pil_to_np(img_PIL, with_transpose=True):
    """
    Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    """
    ar = np.array(img_PIL)
    if len(ar.shape) == 3 and ar.shape[-1] == 4:
        ar = ar[:, :, :3]
        # this is alpha channel
    if with_transpose:
        if len(ar.shape) == 3:
            ar = ar.transpose(2, 0, 1)
        else:
            ar = ar[None, ...]

    return ar.astype(np.float32) / 255.

def data_transforms(img, method=Image.BILINEAR, scale=False):

    ow, oh = img.size
    pw, ph = ow, oh
    if scale == True:
        if ow < oh:
            ow = 512
            oh = ph / pw * 512
        else:
            oh = 512
            ow = pw / ph * 512

    h = int(round(oh / 4) * 4)
    w = int(round(ow / 4) * 4)

    if (h == ph) and (w == pw):
        return img

    return img.resize((w, h), method)

def data_transforms_rgb_old(img):
    w, h = img.size
    A = img
    if w < 256 or h < 256:
        A = transforms.Scale(256, Image.BILINEAR)(img)
    return transforms.CenterCrop(256)(A)

def irregular_hole_synthesize(img, mask):

    img_np = np.array(img).astype("uint8")
    mask_np = np.array(mask).astype("uint8")
    mask_np = mask_np / 255
    img_new = img_np * (1 - mask_np) + mask_np * 255

    hole_img = Image.fromarray(img_new.astype("uint8")).convert("RGB")

    return hole_img

def parameter_set(opt):
    ## Default parameters
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.label_nc = 0
    opt.n_downsample_global = 3
    opt.mc = 64
    opt.k_size = 4
    opt.start_r = 1
    opt.mapping_n_block = 6
    opt.map_mc = 512
    opt.no_instance = True
    opt.checkpoints_dir = "./checkpoints/restoration"
    ##

    if opt.Quality_restore:
        opt.name = "mapping_quality"
        opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "VAE_A_quality")
        #opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "VAE_B_quality")
    if opt.Scratch_and_Quality_restore:
        opt.NL_res = True
        opt.use_SN = True
        opt.correlation_renormalize = True
        opt.NL_use_mask = True
        opt.NL_fusion_method = "combine"
        opt.non_local = "Setting_42"
        opt.name = "mapping_scratch"
        opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "VAE_A_quality")
        opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "VAE_B_scratch")


if __name__ == "__main__":

    opt = TestOptions().parse(save=False)
    parameter_set(opt)

    model = Pix2PixHDModel()

    model.initialize(opt)
    model.eval()

    if not os.path.exists(opt.outputs_dir + "/" + "urhi"):
        os.makedirs(opt.outputs_dir + "/" + "urhi")

    dataset_size = 0

    input_loader = os.listdir(opt.test_input)
    dataset_size = len(input_loader)
    input_loader.sort()

    if opt.test_mask != "":
        mask_loader = os.listdir(opt.test_mask)
        dataset_size = len(os.listdir(opt.test_mask))
        mask_loader.sort()

    img_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    mask_transform = transforms.ToTensor()

    #psnr_sum = 0
    #ssim_sum = 0

    for i in range(dataset_size):

        input_name = input_loader[i]
        input_file = os.path.join(opt.test_input, input_name)
        if not os.path.isfile(input_file):
            print("Skipping non-file %s" % input_name)
            continue
        input = Image.open(input_file).convert("RGB")
        #gt_name = input_name[0:8]+'.jpg'
        #gt_file = os.path.join('/home/dsplxp/code/BOPBTL/test_images/gt/', gt_name)
        #gt = Image.open(gt_file).convert("RGB")

        #print("Now you are processing %s" % (input_name))
        #print(input.size)

        if opt.NL_use_mask:
            mask_name = mask_loader[i]
            mask = Image.open(os.path.join(opt.test_mask, mask_name)).convert("RGB")
            origin = input
            input = irregular_hole_synthesize(input, mask)
            mask = mask_transform(mask)
            mask = mask[:1, :, :]  ## Convert to single channel
            mask = mask.unsqueeze(0)
            input = img_transform(input)
            input = input.unsqueeze(0)
        else:
            if opt.test_mode == "Scale":
                input = data_transforms(input, scale=True)
            if opt.test_mode == "Full":
                input = data_transforms(input, scale=False)
                #gt = data_transforms(gt, scale=False)
            if opt.test_mode == "Crop":
                input = data_transforms_rgb_old(input)
            origin = input
            input = img_transform(input)
            input = input.unsqueeze(0)
            mask = torch.zeros_like(input)
        ### Necessary input
        #print(input.shape)

        try:
            generated = model.inference(input, mask)
        except Exception as ex:
            print("Skip %s due to an error:\n%s" % (input_name, str(ex)))
            continue

        # calculate the ssim bwtween generated and gt
        #print(gt.size)
        #gt_nump = pil_to_np(gt)
        #gt_nump2 = gt_nump.transpose(1, 2, 0)
        gene_nump = ((generated.data.cpu() + 1.0) / 2.0).numpy()
        gene_nump2 = gene_nump[0,:,:,:]
        gene_nump3 = gene_nump2.transpose(1, 2, 0)
        gene_nump4 = gene_nump3[:, :, ::-1]
        #print(gt_nump2)
        #print(compare_ssim(gt_nump2,gene_nump3,multichannel=True))
        #print(compare_ssim(gt_nump2, gene_nump3, multichannel=True))
        #print(compare_psnr(gt_nump2, gene_nump3))
        #psnr_sum = psnr_sum + compare_psnr(gt_nump2, gene_nump3)
        #ssim_sum = ssim_sum + compare_ssim(gt_nump2, gene_nump3, multichannel=True)
        # cv2.imwrite('abagt.jpg',(gt_nump2*255.0).astype(int))
        cv2.imwrite(opt.outputs_dir + '/urhi/' + input_name, (gene_nump4*255).astype(int))
