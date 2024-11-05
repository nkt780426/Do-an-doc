import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler


class RandomResizedCropRect(object):
    def __init__(self, size, scale=(0.8, 1.0)):
        self.size = size
        self.scale = scale

    def __call__(self, img):
        # Lấy kích thước ảnh
        img_width, img_height = img.size

        # Tính toán kích thước và tọa độ cho crop
        crop_width = int(img_width * np.random.uniform(*self.scale))
        crop_height = int(img_height * np.random.uniform(*self.scale))
        crop_left = np.random.randint(0, img_width - crop_width + 1)
        crop_top = np.random.randint(0, img_height - crop_height + 1)
        crop_right = crop_left + crop_width
        crop_bottom = crop_top + crop_height

        # Crop ảnh
        img = img.crop((crop_left, crop_top, crop_right, crop_bottom))

        # Resize ảnh về kích thước mong muốn
        img = img.resize(self.size, resample=Image.BILINEAR)

        return img

class GaussianNoise(object):
    def __init__(self, mean=0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img_np = np.array(img, dtype=np.float32)
        row, col, ch = img_np.shape

        # Tính toán kích thước của vùng ảnh sẽ nhận nhiễu
        min_size = int(row * col * 0.1)
        max_size = int(row * col * 0.25)
        area_size = np.random.randint(min_size, max_size)

        # Tạo ma trận mask để chỉ định vùng của ảnh sẽ nhận nhiễu
        mask = np.zeros((row, col), dtype=np.uint8)
        x = np.random.randint(0, col)
        y = np.random.randint(0, row)
        x_end = min(x + int(np.sqrt(area_size)), col)
        y_end = min(y + int(np.sqrt(area_size)), row)
        mask[y:y_end, x:x_end] = 1

        # Tạo ma trận nhiễu Gaussian
        std = np.random.uniform(0, 0.1)
        gauss = np.random.normal(self.mean, std, (row, col, ch))

        # Áp dụng nhiễu vào phần của ảnh được chỉ định bởi mask
        noisy_img = img_np + gauss * mask[:, :, np.newaxis]
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

        return Image.fromarray(noisy_img)

def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

class SiameseDataset(Dataset):
    def __init__(self, data_dir, transform=None, train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train

        self.image_paths = []
        self.labels = []

        for label in os.listdir(data_dir):
            label_dir = os.path.join(data_dir, label)
            if os.path.isdir(label_dir):
                for image_name in os.listdir(label_dir):
                    image_path = os.path.join(label_dir, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(int(label))  # assuming labels are integers

        self.labels = np.array(self.labels)
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}

        if not self.train:
            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.labels[i]]),
                               1]
                              for i in range(0, len(self.image_paths), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.labels[i]]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.image_paths), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1_path = self.image_paths[index]
            label1 = self.labels[index]
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2_path = self.image_paths[siamese_index]
        else:
            img1_path = self.image_paths[self.test_pairs[index][0]]
            img2_path = self.image_paths[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target
    
    
class TripletDataset(Dataset):
    def __init__(self, data_dir, transform=None, train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train

        self.image_paths = []
        self.labels = []

        for label in os.listdir(data_dir):
            label_dir = os.path.join(data_dir, label)
            if os.path.isdir(label_dir):
                for image_name in os.listdir(label_dir):
                    image_path = os.path.join(label_dir, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(int(label))  # assuming labels are integers

        self.labels = np.array(self.labels)
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}

        if not self.train:
            random_state = np.random.RandomState(29)

            self.test_triplets = [[i,
                                   random_state.choice(self.label_to_indices[self.labels[i]]),
                                   random_state.choice(self.label_to_indices[
                                       np.random.choice(list(self.labels_set - set([self.labels[i]])))
                                   ])
                                  ]
                                 for i in range(len(self.image_paths))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if self.train:
            img1_path = self.image_paths[index]
            label1 = self.labels[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2_path = self.image_paths[positive_index]
            img3_path = self.image_paths[negative_index]
        else:
            img1_path = self.image_paths[self.test_triplets[index][0]]
            img2_path = self.image_paths[self.test_triplets[index][1]]
            img3_path = self.image_paths[self.test_triplets[index][2]]

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        img3 = Image.open(img3_path).convert('RGB')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return (img1, img2, img3), []

class SiameseDatasetConcat(Dataset):
    def __init__(self, data_dir1, data_dir2, transform=None, train=True):
        self.data_dir1 = data_dir1
        self.data_dir2 = data_dir2
        self.transform = transform
        self.train = train

        self.image_paths = []
        self.labels = []

        for label in os.listdir(data_dir1):
            label_dir1 = os.path.join(data_dir1, label)
            label_dir2 = os.path.join(data_dir2, label)
            if os.path.isdir(label_dir1):
                for image_name in os.listdir(label_dir1):
                    image_path1 = os.path.join(label_dir1, image_name)
                    image_path2 = os.path.join(label_dir2, image_name)
                    self.image_paths.append((image_path1, image_path2))
                    self.labels.append(int(label))  # assuming labels are integers

        self.labels = np.array(self.labels)
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}

        if not self.train:
            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.labels[i]]),
                               1]
                              for i in range(0, len(self.image_paths), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.labels[i]]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.image_paths), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1_path = self.image_paths[index]
            label1 = self.labels[index]
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2_path = self.image_paths[siamese_index]
        else:
            img1_path = self.image_paths[self.test_pairs[index][0]]
            img2_path = self.image_paths[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = Image.open(img1_path[0]).convert('RGB')
        img2 = Image.open(img1_path[1]).convert('RGB')        
        img3 = Image.open(img2_path[0]).convert('RGB')
        img4 = Image.open(img2_path[1]).convert('RGB')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            img4 = self.transform(img4)
        return (img1, img2, img3, img4), target
    
    
class TripletDatasetConcat(Dataset):
    def __init__(self, data_dir1, data_dir2, transform=None, train=True):
        self.data_dir1 = data_dir1
        self.data_dir2 = data_dir2
        self.transform = transform
        self.train = train

        self.image_paths = []
        self.labels = []

        for label in os.listdir(data_dir1):
            label_dir1 = os.path.join(data_dir1, label)
            label_dir2 = os.path.join(data_dir2, label)
            if os.path.isdir(label_dir1):
                for image_name in os.listdir(label_dir1):
                    image_path1 = os.path.join(label_dir1, image_name)
                    image_path2 = os.path.join(label_dir2, image_name)
                    self.image_paths.append((image_path1, image_path2))
                    self.labels.append(int(label))  # assuming labels are integers
                    
        self.labels = np.array(self.labels)
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        if not self.train:
            random_state = np.random.RandomState(29)

            self.test_triplets = [[i,
                                   random_state.choice(self.label_to_indices[self.labels[i]]),
                                   random_state.choice(self.label_to_indices[
                                       np.random.choice(list(self.labels_set - set([self.labels[i]])))
                                   ])
                                  ]
                                 for i in range(len(self.image_paths))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if self.train:
            img1_path = self.image_paths[index]
            label1 = self.labels[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2_path = self.image_paths[positive_index]
            img3_path = self.image_paths[negative_index]
        else:
            img1_path = self.image_paths[self.test_triplets[index][0]]
            img2_path = self.image_paths[self.test_triplets[index][1]]
            img3_path = self.image_paths[self.test_triplets[index][2]]

        img1 = Image.open(img1_path[0]).convert('RGB')
        img2 = Image.open(img1_path[1]).convert('RGB')
        img3 = Image.open(img2_path[0]).convert('RGB')
        img4 = Image.open(img2_path[1]).convert('RGB')
        img5 = Image.open(img3_path[0]).convert('RGB')
        img6 = Image.open(img3_path[1]).convert('RGB')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            img4 = self.transform(img4)
            img5 = self.transform(img5)
            img6 = self.transform(img6)
        return (img1, img2, img3, img4, img5, img6), []