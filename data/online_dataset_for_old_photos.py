# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os.path
import io
import zipfile
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from data.Load_Bigfile import BigFileMemoryLoader
import random
import cv2
from io import BytesIO


def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def np_to_pil(img_np):
    '''Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def synthesize_salt_pepper(image, amount, salt_vs_pepper):
    ## Give PIL, return the noisy PIL

    img_pil = pil_to_np(image)

    out = img_pil.copy()
    p = amount
    q = salt_vs_pepper
    flipped = np.random.choice([True, False], size=img_pil.shape,
                               p=[p, 1 - p])
    salted = np.random.choice([True, False], size=img_pil.shape,
                              p=[q, 1 - q])
    peppered = ~salted
    out[flipped & salted] = 1
    out[flipped & peppered] = 0.
    noisy = np.clip(out, 0, 1).astype(np.float32)

    return np_to_pil(noisy)


def synthesize_gaussian(image, std_l, std_r):
    ## Give PIL, return the noisy PIL

    img_pil = pil_to_np(image)

    mean = 0
    std = random.uniform(std_l / 255., std_r / 255.)
    gauss = np.random.normal(loc=mean, scale=std, size=img_pil.shape)
    noisy = img_pil + gauss
    noisy = np.clip(noisy, 0, 1).astype(np.float32)

    return np_to_pil(noisy)


def synthesize_speckle(image, std_l, std_r):
    ## Give PIL, return the noisy PIL

    img_pil = pil_to_np(image)

    mean = 0
    std = random.uniform(std_l / 255., std_r / 255.)
    gauss = np.random.normal(loc=mean, scale=std, size=img_pil.shape)
    noisy = img_pil + gauss * img_pil
    noisy = np.clip(noisy, 0, 1).astype(np.float32)

    return np_to_pil(noisy)


def synthesize_low_resolution(img):
    w, h = img.size

    new_w = random.randint(int(w / 2), w)
    new_h = random.randint(int(h / 2), h)

    img = img.resize((new_w, new_h), Image.BICUBIC)

    if random.uniform(0, 1) < 0.5:
        img = img.resize((w, h), Image.NEAREST)
    else:
        img = img.resize((w, h), Image.BILINEAR)

    return img


def convertToJpeg(im, quality):
    with BytesIO() as f:
        im.save(f, format='JPEG', quality=quality)
        f.seek(0)
        return Image.open(f).convert('RGB')


def blur_image_v2(img):
    x = np.array(img)
    kernel_size_candidate = [(3, 3), (5, 5), (7, 7)]
    kernel_size = random.sample(kernel_size_candidate, 1)[0]
    std = random.uniform(1., 5.)

    # print("The gaussian kernel size: (%d,%d) std: %.2f"%(kernel_size[0],kernel_size[1],std))
    blur = cv2.GaussianBlur(x, kernel_size, std)

    return Image.fromarray(blur.astype(np.uint8))


# def online_add_degradation_v2(img):
#
#     task_id=np.random.permutation(4)
#
#     for x in task_id:
#         if x==0 and random.uniform(0,1)<0.7:
#             img = blur_image_v2(img)
#         if x==1 and random.uniform(0,1)<0.7:
#             flag = random.choice([1, 2, 3])
#             if flag == 1:
#                 img = synthesize_gaussian(img, 5, 50)
#             if flag == 2:
#                 img = synthesize_speckle(img, 5, 50)
#             if flag == 3:
#                 img = synthesize_salt_pepper(img, random.uniform(0, 0.01), random.uniform(0.3, 0.8))
#         if x==2 and random.uniform(0,1)<0.7:
#             img=synthesize_low_resolution(img)
#
#         if x==3 and random.uniform(0,1)<0.7:
#             img=convertToJpeg(img,random.randint(40,100))
#
#     return img
def online_add_degradation_v2(img):
    img_haze = img.crop((0, 0, 256, 256))
    img_clean = img.crop((256, 0, 512, 256))
    return img_haze, img_clean


def irregular_hole_synthesize(img, mask):
    img_np = np.array(img).astype('uint8')
    mask_np = np.array(mask).astype('uint8')
    mask_np = mask_np / 255
    img_new = img_np * (1 - mask_np) + mask_np * 255

    hole_img = Image.fromarray(img_new.astype('uint8')).convert("RGB")

    return hole_img, mask.convert("L")


def zero_mask(size):
    x = np.zeros((size, size, 3)).astype('uint8')
    mask = Image.fromarray(x).convert("RGB")
    return mask


class UnPairOldPhotos_SR(BaseDataset):  ## Synthetic + Real Old
    def initialize(self, opt):
        self.opt = opt
        print(opt.name)
        self.isImage = 'domainA' in opt.name
        self.task = 'old_photo_restoration_training_vae'
        self.dir_AB = opt.dataroot

        ####################################################################################################
        transform_list = []
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))
        self.transform1 = transforms.Compose(transform_list)
        ####################################################################################################

        if self.isImage:

            self.load_img_dir_L_old = os.path.join(self.dir_AB, "Real_L_old.bigfile")
            self.load_img_dir_RGB_old = os.path.join(self.dir_AB, "Real_RGB_old.bigfile")
            self.load_img_dir_clean = os.path.join(self.dir_AB, "VOC_RGB_JPEGImages.bigfile")

            self.loaded_imgs_L_old = BigFileMemoryLoader(self.load_img_dir_L_old)
            self.loaded_imgs_RGB_old = BigFileMemoryLoader(self.load_img_dir_RGB_old)
            self.loaded_imgs_clean = BigFileMemoryLoader(self.load_img_dir_clean)

        else:
            # self.load_img_dir_clean=os.path.join(self.dir_AB,self.opt.test_dataset)
            self.load_img_dir_clean = os.path.join(self.dir_AB, "VOC_RGB_JPEGImages.bigfile")
            self.loaded_imgs_clean = BigFileMemoryLoader(self.load_img_dir_clean)

        ####
        print("-------------Filter the imgs whose size <256 in VOC-------------")
        self.filtered_imgs_clean = []
        for i in range(len(self.loaded_imgs_clean)):
            img_name, img = self.loaded_imgs_clean[i]
            h, w = img.size
            if h < 256 or w < 256:
                continue
            self.filtered_imgs_clean.append((img_name, img))
        #            img2 = img.crop((0,0,400,400))--------------------------------------------------------------------------------------
        #            print(img2.size)-------------------------------------------------------------------------------------------------

        print("--------Origin image num is [%d], filtered result is [%d]--------" % (
            len(self.loaded_imgs_clean), len(self.filtered_imgs_clean)))
        ## Filter these images whose size is less than 256

        # self.img_list=os.listdir(load_img_dir)
        self.pid = os.getpid()

    def __getitem__(self, index):

        sampled_dataset_clean = None
        sampled_dataset_real = None
        if self.isImage:  ## domain A , contains 2 kinds of data: synthetic + real_old
            #            if random.uniform(0,1)<0.5:
            #                sampled_dataset_real=self.loaded_imgs_L_old
            #                self.load_img_dir_real=self.load_img_dir_L_old
            #            else:
            sampled_dataset_real = self.loaded_imgs_RGB_old
            self.load_img_dir_real = self.load_img_dir_RGB_old

            sampled_dataset_clean = self.filtered_imgs_clean
            self.load_img_dir_clean = self.load_img_dir_clean
        else:
            sampled_dataset = self.filtered_imgs_clean
            self.load_img_dir = self.load_img_dir_clean

        sampled_dataset_len = len(sampled_dataset_clean)
        sampled_dataset_len_real = len(sampled_dataset_real)
        index = random.randint(0, sampled_dataset_len - 1)
        index_real = random.randint(0, sampled_dataset_len_real - 1)
        img_name_clean, img_clean = sampled_dataset_clean[index]
        img_name_real, img_real = sampled_dataset_real[index_real]

        ##################################################################################################

        syn_density_path = os.path.join('/data/lxp/haze/dataset/syn_density_abs', img_name_clean)
        syn_density = Image.open(syn_density_path)

        path2 = os.path.join('/data/lxp/haze/dataset2/syn_comb', img_name_clean)
        img_clean2 = Image.open(path2).convert('RGB')
        density_path2 = os.path.join('/data/lxp/haze/dataset2/syn_density_abs', img_name_clean)
        syn_density2 = Image.open(density_path2)

        path3 = os.path.join('/data/lxp/haze/dataset3/syn_comb', img_name_clean)
        img_clean3 = Image.open(path3).convert('RGB')
        density_path3 = os.path.join('/data/lxp/haze/dataset3/syn_density_abs', img_name_clean)
        syn_density3 = Image.open(density_path3)

        touzi = random.uniform(0, 1)
        if touzi <= 0.33:
            img_clean = img_clean2
            syn_density = syn_density2
        elif touzi >= 0.66:
            img_clean = img_clean3
            syn_density = syn_density3

        density_haze, density_clean = online_add_degradation_v2(syn_density)
        density_haze = self.transform1(density_haze)
        density_clean = self.transform1(density_clean)

        real_density_path = os.path.join('/data/lxp/haze/dataset/Real_density_abs', img_name_real)
        real_density = Image.open(real_density_path)
        density_real = self.transform1(real_density)

        ##################################################################################################

        #        print(img.size)
        img_haze, image_clean = online_add_degradation_v2(img_clean)

        path = os.path.join(self.load_img_dir_clean, img_name_clean)

        path_real = os.path.join(self.load_img_dir_real, img_name_real)

        # AB = Image.open(path).convert('RGB')
        # split AB image into A and B

        # apply the same transform to both A and B

        # if random.uniform(0,1) <0.1:
        #    img=img.convert("L")
        #     img=img.convert("RGB")
        ## Give a probability P, we convert the RGB image into L

        A = img_haze
        B = image_clean
        C = img_real
        w, h = A.size
        w_C, h_C = C.size

        if w < 256 or h < 256:
            A = transforms.Resize(256, Image.BICUBIC)(A)
            B = transforms.Resize(256, Image.BICUBIC)(B)
        if w_C < 256 or h_C < 256:
            C = transforms.Resize(256, Image.BICUBIC)(C)
        ## Since we want to only crop the images (256*256), for those old photos whose size is smaller than 256, we first resize them.

        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params)
        B_transform = get_transform(self.opt, transform_params)
        A_tensor = A_transform(A)
        B_tensor = B_transform(B)

        transform_params_C = get_params(self.opt, C.size)
        C_transform = get_transform(self.opt, transform_params_C)
        C_tensor = C_transform(C)

        inst_tensor = feat_tensor = 0

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor,
                      'feat': feat_tensor, 'real': C_tensor, 'path': path, 'path_real': path_real,
                      'density_syn': density_haze, 'density_gt': density_clean, 'density_real': density_real}
        return input_dict

    def __len__(self):
        return len(
            self.loaded_imgs_clean)  ## actually, this is useless, since the selected index is just a random number

    def name(self):
        return 'UnPairOldPhotos_SR'


class PairOldPhotos(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.isImage = 'imagegan' in opt.name
        self.task = 'old_photo_restoration_training_mapping'
        self.dir_AB = opt.dataroot
        if opt.isTrain:
            self.load_img_dir_clean = os.path.join(self.dir_AB, "VOC_RGB_JPEGImages.bigfile")
            self.loaded_imgs_clean = BigFileMemoryLoader(self.load_img_dir_clean)

            print("-------------Filter the imgs whose size <256 in VOC-------------")
            self.filtered_imgs_clean = []
            for i in range(len(self.loaded_imgs_clean)):
                img_name, img = self.loaded_imgs_clean[i]
                h, w = img.size
                if h < 256 or w < 256:
                    continue
                self.filtered_imgs_clean.append((img_name, img))

            print("--------Origin image num is [%d], filtered result is [%d]--------" % (
                len(self.loaded_imgs_clean), len(self.filtered_imgs_clean)))

        else:
            self.load_img_dir = os.path.join(self.dir_AB, opt.test_dataset)
            self.loaded_imgs = BigFileMemoryLoader(self.load_img_dir)

        self.pid = os.getpid()

    def __getitem__(self, index):

        if self.opt.isTrain:
            img_name_clean, BB = self.filtered_imgs_clean[index]
            path = os.path.join(self.load_img_dir_clean, img_name_clean)
            if self.opt.use_v2_degradation:
                A = online_add_degradation_v2(BB)
            B = BB.crop((0, 256, 256, 512))
            print(B.size)
            ### Remind: A is the input and B is corresponding GT
        else:

            if self.opt.test_on_synthetic:

                img_name_B, B = self.loaded_imgs[index]
                A = online_add_degradation_v2(B)
                img_name_A = img_name_B
                path = os.path.join(self.load_img_dir, img_name_A)
            else:
                img_name_A, A = self.loaded_imgs[index]
                img_name_B, B = self.loaded_imgs[index]
                path = os.path.join(self.load_img_dir, img_name_A)

        if random.uniform(0, 1) < 0.1 and self.opt.isTrain:
            A = A.convert("L")
            B = B.convert("L")
            A = A.convert("RGB")
            B = B.convert("RGB")
        ## In P, we convert the RGB into L

        ##test on L

        # split AB image into A and B
        # w, h = img.size
        # w2 = int(w / 2)
        # A = img.crop((0, 0, w2, h))
        # B = img.crop((w2, 0, w, h))
        w, h = A.size
        if w < 256 or h < 256:
            A = transforms.Resize(256, Image.BICUBIC)(A)
            B = transforms.Resize(256, Image.BICUBIC)(B)

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params)
        B_transform = get_transform(self.opt, transform_params)

        B_tensor = inst_tensor = feat_tensor = 0
        A_tensor = A_transform(A)
        B_tensor = B_transform(B)

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor,
                      'feat': feat_tensor, 'path': path}
        return input_dict

    def __len__(self):

        if self.opt.isTrain:
            return len(self.filtered_imgs_clean)
        else:
            return len(self.loaded_imgs)

    def name(self):
        return 'PairOldPhotos'


class PairOldPhotos_with_hole(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.isImage = 'imagegan' in opt.name
        self.task = 'old_photo_restoration_training_mapping'
        self.dir_AB = opt.dataroot
        if opt.isTrain:
            self.load_img_dir_clean = os.path.join(self.dir_AB, "VOC_RGB_JPEGImages.bigfile")
            self.loaded_imgs_clean = BigFileMemoryLoader(self.load_img_dir_clean)

            print("-------------Filter the imgs whose size <256 in VOC-------------")
            self.filtered_imgs_clean = []
            for i in range(len(self.loaded_imgs_clean)):
                img_name, img = self.loaded_imgs_clean[i]
                h, w = img.size
                if h < 256 or w < 256:
                    continue
                self.filtered_imgs_clean.append((img_name, img))

            print("--------Origin image num is [%d], filtered result is [%d]--------" % (
                len(self.loaded_imgs_clean), len(self.filtered_imgs_clean)))

        else:
            self.load_img_dir = os.path.join(self.dir_AB, opt.test_dataset)
            self.loaded_imgs = BigFileMemoryLoader(self.load_img_dir)

        self.loaded_masks = BigFileMemoryLoader(opt.irregular_mask)

        self.pid = os.getpid()

    def __getitem__(self, index):

        if self.opt.isTrain:
            img_name_clean, B = self.filtered_imgs_clean[index]
            path = os.path.join(self.load_img_dir_clean, img_name_clean)

            B = transforms.RandomCrop(256)(B)
            A = online_add_degradation_v2(B)
            ### Remind: A is the input and B is corresponding GT

        else:
            img_name_A, A = self.loaded_imgs[index]
            img_name_B, B = self.loaded_imgs[index]
            path = os.path.join(self.load_img_dir, img_name_A)

            # A=A.resize((256,256))
            A = transforms.CenterCrop(256)(A)
            B = A

        if random.uniform(0, 1) < 0.1 and self.opt.isTrain:
            A = A.convert("L")
            B = B.convert("L")
            A = A.convert("RGB")
            B = B.convert("RGB")
        ## In P, we convert the RGB into L

        if self.opt.isTrain:
            mask_name, mask = self.loaded_masks[random.randint(0, len(self.loaded_masks) - 1)]
        else:
            mask_name, mask = self.loaded_masks[index % 100]
        mask = mask.resize((self.opt.loadSize, self.opt.loadSize), Image.NEAREST)

        if self.opt.random_hole and random.uniform(0, 1) > 0.5 and self.opt.isTrain:
            mask = zero_mask(256)

        if self.opt.no_hole:
            mask = zero_mask(256)

        A, _ = irregular_hole_synthesize(A, mask)

        if not self.opt.isTrain and self.opt.hole_image_no_mask:
            mask = zero_mask(256)

        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params)
        B_transform = get_transform(self.opt, transform_params)

        if transform_params['flip'] and self.opt.isTrain:
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        mask_tensor = transforms.ToTensor()(mask)

        B_tensor = inst_tensor = feat_tensor = 0
        A_tensor = A_transform(A)
        B_tensor = B_transform(B)

        input_dict = {'label': A_tensor, 'inst': mask_tensor[:1], 'image': B_tensor,
                      'feat': feat_tensor, 'path': path}
        return input_dict

    def __len__(self):

        if self.opt.isTrain:
            return len(self.filtered_imgs_clean)

        else:
            return len(self.loaded_imgs)

    def name(self):
        return 'PairOldPhotos_with_hole'