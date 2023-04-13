import os
from enum import Enum

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# Image_MEAN = [0.3957,] #　only for class 2
# Image_STD = [0.1276,]



class DAGMDataset(Dataset):

    def __init__(
            self,
            source,
            classname,
            mean=0.5,
            std=0.5,
            resize=256,
            imagesize=224,
            split='train',
            **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.classname = classname

        self.data_to_iterate = self.get_image_data()

        self.transform_img = transforms.Compose([transforms.Resize(resize),
                                                 transforms.RandomCrop(imagesize),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=mean, std=std)
                                                ] if split == 'train'
                                           else [transforms.Resize(resize),
                                                 transforms.CenterCrop(imagesize),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=mean, std=std)
                                                ])

        self.transform_mask = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ])
        self.imagesize = (1, imagesize, imagesize)
        self.transform_mean = mean
        self.transform_std = std

    def __getitem__(self, idx):
        label, image_path, mask_path, name = self.data_to_iterate[idx]
        image = Image.open(image_path).convert("L")
        image = self.transform_img(image)

        if self.split == 'test' and mask_path is not None:
            mask = Image.open(mask_path).convert('L')
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros(self.imagesize)

        return {
            "image": image,
            'name':name,
            "mask": mask,
            "label": label
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        data_to_iterate = None
        if self.split in ('train', 'val'):
            classpath = os.path.join(self.source, self.classname, self.split)
            imgs = os.listdir(classpath)
            imgs = [os.path.join(classpath, x) for x in imgs if x.endswith('.PNG')]
            data_to_iterate = [[None for _ in range(4)] for _ in range(len(imgs))]  #四元组为（label, imgpath, maskpath, name）
            for _ in range(len(imgs)):
                data_to_iterate[_][0] = 0
                data_to_iterate[_][1] = imgs[_]
                data_to_iterate[_][3] = os.path.split(imgs[_])[-1]
        elif self.split == 'test':
            classpath = os.path.join(self.source, self.classname, self.split)
            types = os.listdir(classpath)
            rec = {}
            for typ in types:
                if typ != 'mask':
                    imgpaths = os.listdir(os.path.join(classpath, typ))
                    imgs = [os.path.join(classpath, typ, x) for x in imgpaths if x.endswith('.PNG')]

                    rec[typ] = [[None for _ in range(4)] for _ in range(len(imgs))]
                    for _ in range(len(imgs)):
                        img_name = os.path.split(imgs[_])[-1]
                        rec[typ][_][0] = 0
                        rec[typ][_][1] = imgs[_]
                        rec[typ][_][3] = img_name
                        if typ == 'defect':
                            rec[typ][_][0] = 1
                            mask_name = img_name[:4] + '_label.PNG'
                            mask_path = os.path.join(classpath, 'mask', mask_name)
                            rec[typ][_][2] = mask_path
            data_to_iterate = []
            for key in rec.keys():
                data_to_iterate.extend(rec[key])

        else:
            raise KeyError(f"{self.split} not in ('train', 'val', 'test')")

        return data_to_iterate


    
    
class TestDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, resize: int = 256, imagesize: int = 224,
                 mask_suffix: str = '_label.PNG', classname: str = 'class6'):
        self.images_dir = images_dir
        self.mask_dir = mask_dir
        self.mask_suffix = mask_suffix
        self.classname = classname
        self.imagesize = imagesize
        self.imgs = [os.path.split(file)[-1] for file in os.listdir(images_dir) if file.endswith('.PNG')]
        labels = []
        masks = []
        for file in self.imgs:
            mask_path = os.path.join(mask_dir, file[:4]+mask_suffix)
            if os.path.exists(mask_path):
                labels.append(1)
                masks.append(mask_path)
            else:
                labels.append(0)
                masks.append(None)
        # assert len(self.imgs_A) != 0 ,'数据集A为空'
        # assert len(self.imgs_B) != 0 ,'数据集B为空'
        print(f'Creating datasetA with {len(self.imgs) - sum(labels)} examples and datasetsB with {sum(labels)}')
        self.labels = labels
        self.masks = masks

        self.transform_img = transforms.Compose([transforms.Resize(resize),
                                                 transforms.CenterCrop(imagesize),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=0.5, std=0.5)
                                                 ])
        self.transform_mask = transforms.Compose([transforms.Resize(resize),
                                                  transforms.CenterCrop(imagesize),
                                                  transforms.ToTensor(),
                                                  ])

    def __len__(self):
        return  len(self.imgs)


    def __getitem__(self, idx):
        name = self.imgs[idx]
        img = Image.open(os.path.join(self.images_dir, self.imgs[idx])).convert('L')
        img = self.transform_img(img)
        mask_path = self.masks[idx]
        if mask_path:
            mask = Image.open(mask_path).convert('L')
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros((1, self.imagesize, self.imagesize))

        label = self.labels[idx]


        return {
            'image': img,
            'mask': mask,
            'label':label,
            'name':name
        }


