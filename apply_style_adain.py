import torch
from torchvision import models
from helpers import image_loader, save_image
from model_adain import get_model_and_losses

'''
학습한 Neural Style Transfer + AdaIN 모델을 적용한다

'''

# Pytorch GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available() else 256

# 환경 변수
style_img_path = 'panda_images/fubao/fubao_2.jpg'
content_img_path = 'images/bear/bear_1.jpg'
output_img_path = 'output_images/user_image_panda_styled_adain.jpg'
saved_model_path = 'saved_models/panda_style_model_adain.pth'

# 이미지 로드
style_img = image_loader(style_img_path, imsize).to(device)
content_img = image_loader(content_img_path, imsize).to(device)

# 사전 학습된 VGG19 모델 로드
cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# model.py에서 정의한 모델과, Style, Content Loss 클래스 객체 생성
model, style_losses, content_losses = get_model_and_losses(
    cnn, cnn_normalization_mean, cnn_normalization_std, style_img, content_img)

# 사전 학습된 모델 파라미터 로드
model.load_state_dict(torch.load(saved_model_path))

# 입력 이미지 초기화
input_img = content_img.clone()

# Optimizer : AdaIN 방식은 Adam이 가장 적합하다
# SparseAdam은 못씀, ASGD는 아예 별로
optimizer = torch.optim.Adam([input_img.requires_grad_()], lr=0.01)

# 스타일 적용
num_steps = 300
style_weight = 1e6 # 기본은 1e6
content_weight = 1 # 기본은 1

print("Applying panda style...")
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

# 최종 이미지 저장
save_image(input_img, output_img_path)
print("Panda style applied and image saved.")