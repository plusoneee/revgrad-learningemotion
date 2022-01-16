from skimage import io
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import config as cnf

class StudentEmotion(Dataset):
    def __init__(self, root_dir='stu_emotion', csv_file=cnf.STUDENT_ANNOTAION, transform=None):
        self._root = root_dir
        self._transform = transform
        self._annotaions = pd.read_csv(csv_file)

    def __len__(self):
        return len(self._annotaions)
    
    def __getitem__(self, index):
        
        image_path = os.path.join(self._root, self._annotaions['path'][index]) 
        image = io.imread(image_path) 

        label = int(self._annotaions['label'][index])
        label = torch.tensor(label)
        
        if self._transform:
            image = self._transform(image)
        return (image, label)
        
class FER2013(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
    
        self._root = root_dir
        if train:
            self._root = root_dir + os.sep + 'train/'
            self._csv_file = cnf.FER2013_TRAIN_ANNOTATION
        else:
            self._root = root_dir + os.sep + 'test/'
            self._csv_file = cnf.FER2013_TEST_ANNOTATION

        self._transform = transform
        self._annotaions = pd.read_csv(self._csv_file)

    def __len__(self):
        return len(self._annotaions)
    
    def __getitem__(self, index):
        
        
        image_path = os.path.join(self._root,  self._annotaions['path'][index])
        
        image = io.imread(image_path)
        
        label = int(self._annotaions['label'][index])
        label = torch.tensor(label)
        
        # transform
        if self._transform:
            image = self._transform(image)

        return (image, label)


def test_fer2013_dataset():
    from torchvision import transforms
    batch_size = 64
    # transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((cnf.IMAGE_SIZE, cnf.IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    dataset = FER2013('fer2013', train=False, transform=transform)
    dataloader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = False
    )

    for (batch_images, batch_labels) in DataLoader(dataset=dataset, batch_size =32, shuffle=True):
        print('Image Shape:', batch_images.shape) # shape (B, C, H, W)
        print('Label Shape:', batch_labels.shape)
        break

def test_student_emotion_dataset():
    from torchvision import transforms
    batch_size = 64
    # transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((cnf.IMAGE_SIZE, cnf.IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    dataset = StudentEmotion('stu_emotion', transform=transform)
    dataloader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = False
    )
    for (batch_images, batch_labels) in DataLoader(dataset=dataset, batch_size =32, shuffle=True):
        print('Image Shape:', batch_images.shape) # shape (B, C, H, W)
        print('Label Shape:', batch_labels.shape)
        break

if __name__ == '__main__':
    test_student_emotion_dataset()
    test_fer2013_dataset()
    
