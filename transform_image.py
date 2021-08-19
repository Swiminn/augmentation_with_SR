from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord
from test_liif import batched_predict

def bicubic(img_dir, h, w):
    img = Image.open(img_dir).convert('RGB')
    img = img.resize((h, w),Image.BICUBIC)
    return img



def liif(img_dir, h ,w):
    img = transforms.ToTensor()(Image.open(img_dir).convert('RGB'))
    model = models.make(torch.load('rdn-liif.pth')['model'], load_sd=True).cuda()
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
                           coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    img = transforms.ToPILImage()(pred)
    return img


class Upscailing(torch.nn.Module):

    def __init__(self, h, w, method="bicubic"):
        super().__init__()
        self.h = h
        self.w = w
        self.method = method

    def forward(self, img):
        if self.method == "bicubic":
            img = img.resize((self.h, self.w), Image.BICUBIC)
            return img
        elif self.method == "liif":
            model = models.make(torch.load('rdn-liif.pth')['model'], load_sd=True).cuda()
            coord = make_coord((self.h, self.w)).cuda()
            cell = torch.ones_like(coord)
            cell[:, 0] *= 2 / self.h
            cell[:, 1] *= 2 / self.w
            pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
                                   coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
            pred = (pred * 0.5 + 0.5).clamp(0, 1).view(self.h, self.w, 3).permute(2, 0, 1).cpu()
            img = transforms.ToPILImage()(pred)
            return liif(img)
        else:
            print("input method is wrong")

if __name__ == "__main__":
    image1 = liif('2.png',512,512)
    image1.save("test_liif.png")
    image2 = bicubic('2.png', 512,512)
    image2.save("test_bicubic.png")