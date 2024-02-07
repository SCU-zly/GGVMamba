import os
import cv2
import pdb
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image,ImageOps
from torchvision import transforms
from torch.utils.data import Dataset
from einops import rearrange
import argparse


import cv2
from albumentations import ShiftScaleRotate,HorizontalFlip,RandomResizedCrop,Compose,Resize
from albumentations.augmentations.transforms import Lambda, RandomBrightnessContrast
from albumentations.pytorch import ToTensorV2 as ToTensor
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
# ========================================================================
# BASE FUNCTION
def load_json(route_json):
    with open(route_json, 'r') as f:
        data = json.load(f)
    return data
# ========================================================================
# BASE FUNCTION
def crop_img(image, bbox):
    cropped = image[bbox[1]:bbox[1]+bbox[3],
                          bbox[0]:bbox[0]+bbox[2]]
    
    return cropped
# ========================================================================
# FOR FINE-TUNING
# RHPE pre processing
# Hard way to get left hand
def crop_one_hand_anno(image_local_path, image_name, rois):
    img = Image.open(image_local_path)
    # Loading as PIL.Image, converting to numpy while ensuring grayscale
    img = img.convert('L')
    img = np.array(img)
    imgs_list = rois['images']
    img_id=[im['id'] for im in imgs_list if im['file_name']==image_name+'.png'][0]
    del imgs_list

    # Getting the image annotations, using said ID 
    annotations = rois['annotations']

    for im_ann in annotations:
        if im_ann['image_id'] == img_id:
            im_kpts = im_ann['keypoints']
            im_bbox = (list(map(int, im_ann['bbox'])))
            break
    del annotations
    del rois  # Cleaning memory
    cropped_img = crop_img(img, im_bbox)
    #RHPE data are always inverted
    if np.argmax(np.histogram(cropped_img,20)[0])>10:
        invert_crop_img = np.array(ImageOps.invert(Image.fromarray(cropped_img).convert('RGB')))
    else:
        invert_crop_img = cropped_img
        
    return invert_crop_img
# ========================================================================
# FOR FINE-TUNING
# RHPE pre processing
# Easy way to get left hand
def crop_one_hand_half(image_local_path):
    img = Image.open(image_local_path)
    # Loading as PIL.Image, converting to numpy while ensuring grayscale
    img = img.convert('L')
    img = np.array(img)
    w,h=img.shape
    im_bbox=[0,0,w//2,h]
    cropped_img = crop_img(img, im_bbox)
    #RHPE data are always inverted
    if np.argmax(np.histogram(cropped_img,20)[0])>10:
        invert_crop_img = np.array(ImageOps.invert(Image.fromarray(cropped_img).convert('RGB')))
    else:
        invert_crop_img = np.array(Image.fromarray(cropped_img).convert('RGB'))
        
    return invert_crop_img
# ========================================================================

class BAADataset(Dataset):
    """BoneAge dataset for fine-tuning"""
    def __init__(self, img_dir, ann_file, input_size,
                 json_file = None, img_transform = None, dataset=None):
        self.annotations = pd.concat([pd.read_csv(f) for f in ann_file])
        self.img_dir = img_dir
        
        self.kpts = [load_json(f) for f in json_file] if json_file else None
        #TODO
        self.img_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor()]
        ) if img_transform is None else img_transform
    
        self.dataset = dataset if dataset else 'RHPE'

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
        info = self.annotations.iloc[idx]
        if self.dataset == 'RHPE':
            image_name=str(info[0]).zfill(5)
        else:
            image_name=str(info[0])
        
        for i in range(len(self.img_dir)):
            image_local_path = os.path.join(self.img_dir[i], image_name + '.png')
            if self.dataset == 'RSNA':
                if os.path.exists(image_local_path):
                    img = Image.open(image_local_path)
                    # Loading as PIL.Image, converting to numpy while ensuring grayscale
                    img = img.convert('RGB')
                    img = np.array(img) 
                    break    
            elif self.dataset == 'RHPE':
                if os.path.exists(image_local_path):
                    if self.kpts is None:
                        img = crop_one_hand_half(image_local_path)
                    else:
                        img = crop_one_hand_anno(image_local_path, image_name, self.kpts[i])
                    break
        
        #TODO 这尼玛RHPE/RSNA的inference没标签啊... DHA的inference有标签，但是是不是这样写呢？
        bone_age = torch.tensor(info[2], dtype = torch.float)
        gender = torch.tensor(info[1]*1, dtype = torch.float).unsqueeze(-1)

        if self.dataset == 'RHPE':
            # if self.inference:
            #     chronological_age = torch.tensor(info[2], dtype = torch.float).unsqueeze(-1)
            # else:
                chronological_age = torch.tensor(info[3], dtype = torch.float).unsqueeze(-1)
        else:
            chronological_age = torch.tensor(0, dtype = torch.float).unsqueeze(-1)

        if self.img_transform:
            if type(self.img_transform) == Compose:
                
                out_im = self.img_transform(image = img)["image"]
            else:
                out_im = self.img_transform(Image.fromarray(img))

        else:
            out_im = img
        out_im = out_im.to(torch.float32)
        return out_im, gender, chronological_age, bone_age #x,y,z,label

    def __len__(self):
        return len(self.annotations)
    
# ======================================================================

def build_BAA_dataset_train(is_train, args):
    '''
    It is import to init args!!
    first args.input_size
    second args.img_transform
    third args.dataset
    if dataset is not RHPE,then is_train is false 
    '''
    if args.dataset == 'RSNA':
        if is_train:
            img_dir = [
            # "/data2/zly/BoneData/RSNA/DATA_TRAIN",
            "/home/data2/zly/BoneData/RSNA/DATA_TRAIN",
            ]
            ann_file = [
            # "/data2/zly/BoneData/RSNA/ANN_PATH_TRAIN.csv",
            "/home/data2/zly/BoneData/RSNA/ANN_PATH_TRAIN.csv",
            ]
            json_file = [
            # "/data2/zly/BoneData/RSNA/ROIS_PATH_TRAIN.json",
            "/home/data2/zly/BoneData/RSNA/ROIS_PATH_TRAIN.json",
            ]
        else:
            img_dir = [
                # "/data2/zly/BoneData/RSNA/DATA_VAL"
                "/home/data2/zly/BoneData/RSNA/DATA_VAL"
                ]
            ann_file = [
                # "/data2/zly/BoneData/RSNA/ANN_PATH_VAL.csv",
                "/home/data2/zly/BoneData/RSNA/ANN_PATH_VAL.csv",
            ]
            json_file = [
                # "/data2/zly/BoneData/RSNA/ROIS_PATH_VAL.json",
                "/home/data2/zly/BoneData/RSNA/ROIS_PATH_VAL.json",
            ]

        
    #elif in DHA
    else:
        if is_train:
            img_dir =[
                # "/data2/zly/BoneData/RHPE/RHPE_TRAIN"
                "/home/data2/zly/BoneData/RHPE/RHPE_TRAIN"
            ]
            ann_file = [
                # "/data2/zly/BoneData/RHPE/RHPE_ANNOTATIONS/RHPE_Boneage_train.csv"
                "/home/data2/zly/BoneData/RHPE/RHPE_ANNOTATIONS/RHPE_Boneage_train.csv"
            ]
            #TODO whether to use easy crop to augment data
            json_file = [
                # "/data2/zly/BoneData/RHPE/RHPE_ANNOTATIONS/RHPE_anatomical_ROIs_train.json"
                "/home/data2/zly/BoneData/RHPE/RHPE_ANNOTATIONS/RHPE_anatomical_ROIs_train.json"
            ]

        else:
            img_dir =[
                # "/data2/zly/BoneData/RHPE/RHPE_VAL"
                "/home/data2/zly/BoneData/RHPE/RHPE_VAL"
            ]
            ann_file = [
                # "/data2/zly/BoneData/RHPE/RHPE_ANNOTATIONS/RHPE_Boneage_val.csv"
                "/home/data2/zly/BoneData/RHPE/RHPE_ANNOTATIONS/RHPE_Boneage_val.csv"
            ]
            json_file = [
                # "/data2/zly/BoneData/RHPE/RHPE_ANNOTATIONS/RHPE_anatomical_ROIs_val.json"
                "/home/data2/zly/BoneData/RHPE/RHPE_ANNOTATIONS/RHPE_anatomical_ROIs_val.json"
            ]

    dataset = BAADataset(input_size=args.input_size,img_transform=args.img_transform,dataset=args.dataset,img_dir=img_dir, ann_file=ann_file,json_file=json_file)
    return dataset
        
if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser('BAA pre-training test', add_help=False)
        parser.add_argument('--input_size', default=512, type=int,
                            help='images input size for backbone')
        opt = parser.parse_args(args=[])
        RandomErasing = transforms.RandomErasing(scale=(0.02, 0.08), ratio = (0.5, 2), p = 0.8)

        def randomErase(image, **kwargs):
            return RandomErasing(image)

        def sample_normalize(image, **kwargs):
            image = image/255
            channel = image.shape[2]
            mean, std = image.reshape((-1, channel)).mean(axis = 0), image.reshape((-1, channel)).std(axis = 0)
            return (image-mean)/(std + 1e-3)
        transform_train = Compose([
        # RandomBrightnessContrast(p = 0.8),
        Resize(512, 512),
        RandomResizedCrop(512, 512,
                          (0.5, 1.0), p = 0.5),
        ShiftScaleRotate(shift_limit = 0.2, scale_limit = 0.2, rotate_limit=20, border_mode = cv2.BORDER_CONSTANT, value = 0.0, p = 0.8),
        # HorizontalFlip(p = 0.5),
        
        # ShiftScaleRotate(shift_limit = 0.2, scale_limit = 0.2, rotate_limit=20, p = 0.8),
        HorizontalFlip(p = 0.5),
        RandomBrightnessContrast(p = 0.8, contrast_limit=(-0.3, 0.2)),                             
        Lambda(image = sample_normalize),
        ToTensor(),
        Lambda(image = randomErase) 
        
        ])

        transform_val = Compose([                                   
            Lambda(image = sample_normalize),
            ToTensor(),
        ])
        # opt.img_transform =  transforms.Compose([
        #     transforms.Resize((512, 512)),
        #     transforms.ToTensor()]
        # ) 
        opt.img_transform = transform_train
        opt.dataset = 'RSNA'
        is_train = False
        dataset = build_BAA_dataset_train(is_train,opt)

        
        data_loader_train = torch.utils.data.DataLoader(
            dataset,
            batch_size=10,
            num_workers=12,
            # pin_memory=args.pin_mem,
            drop_last=True,
            # worker_init_fn=utils.seed_worker
        )
        for (whats,batch) in enumerate(data_loader_train):
            img, sex, cAge, label = batch
            print(img.shape,sex.shape,cAge.shape,label.shape)
            print(img.dtype,sex.dtype,cAge.dtype,label.dtype)
            break
    except Exception as e:
        print(e)
        import pdb; pdb.set_trace()