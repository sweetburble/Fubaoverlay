import torch
import torch.optim as optim
from torchvision import models
from helpers import image_loader, save_image
from model import get_model_and_losses

'''
(다양한 크기의 입력 이미지를 가지고) Neural Style Transfer 모델을 학습한다

'''

# Pytorch GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 환경 변수
imsize_list = [128, 256, 512]  # 학습시킬 다양한 크기의 이미지 리스트
style_img_path = 'panda_images/fubao/fubao_2.jpg'
content_img_path = 'images/bear/bear_2.jpg'
output_model_path = 'saved_models/panda_style_model_multiscale.pth'

# 이미지 로드
style_img = image_loader(style_img_path, max(imsize_list)).to(device)

# 사전 학습된 VGG19 모델 로드
cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# 다양한 크기에서 학습 시작
input_img = None
for imsize in imsize_list:
    content_img = image_loader(content_img_path, imsize).to(device)
    model, style_losses, content_losses = get_model_and_losses(
        cnn, cnn_normalization_mean, cnn_normalization_std, style_img, content_img)

    if input_img is None:
        input_img = content_img.clone()
    else:
        input_img = torch.nn.functional.interpolate(input_img, size=(imsize, imsize), mode='bilinear')

    optimizer = optim.LBFGS([input_img.requires_grad_()])

    num_steps = 300
    style_weight = 1e6
    content_weight = 1

    print(f'Optimizing at size {imsize}...')
    run = [0]
    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = sum([sl.loss for sl in style_losses])
            content_score = sum([cl.loss for cl in content_losses])

            loss = style_weight * style_score + content_weight * content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Run {run[0]}:")
                print(f"Style Loss : {style_score.item():4f} Content Loss: {content_score.item():4f}")

            return style_weight * style_score + content_weight * content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)

# 최종 이미지와 학습된 모델 저장
save_image(input_img, 'output_images/panda_styled_image_multiscale.jpg')
torch.save(model.state_dict(), output_model_path)
print("Multi-scale training completed and model saved.")