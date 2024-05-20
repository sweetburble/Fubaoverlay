from PIL import Image
import torch
from torchvision import transforms
import os

'''
다양한 유틸리티 함수들
'''

# 이미지 로딩 및 변환 함수
def image_loader(image_name, imsize):
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),  # 이미지가 같은 크기로 조정되도록 함
        transforms.ToTensor()
    ])
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(torch.float)

# 스타일 이미지 로딩 함수
def load_style_images(style_dir, imsize):
    style_images = []
    for filename in os.listdir(style_dir):
        if filename.endswith(('jpg', 'jpeg', 'png')):
            style_images.append(image_loader(os.path.join(style_dir, filename), imsize))
    return torch.cat(style_images, 0)

# 이미지 유틸리티 저장 함수
def save_image(tensor, output_path):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()  
    image = image.squeeze(0)      
    image = unloader(image)
    image.save(output_path)