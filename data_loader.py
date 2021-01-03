import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import torch


# this dataset contains around 125.000 images
def get_data_loader(opt):
    dataloader = torch.utils.data.DataLoader(
            datasets.LSUN(
                root="./data",
                classes=['church_outdoor_train'],
                transform=transforms.Compose([
                            transforms.Resize((opt.img_size , opt.img_size)),
                            transforms.ToTensor()
                            ])
            ),
            batch_size=opt.batch_size,
            shuffle=True,
        )
    return dataloader