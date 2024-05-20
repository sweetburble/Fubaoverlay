import torch
import torch.optim as optim
from torchvision import models
from torchvision.models import VGG19_Weights
from helpers import image_loader, save_image
from model_adain import get_model_and_losses

'''
Neural Style Transfer 모델을 학습한다 + AdaIN 적용

컴퓨터 비전 분야에서 Adam은 딥러닝 모델 학습에 널리 사용되는 최적화 알고리즘입니다.

Adam의 주요 특징:

Adaptive Moment Estimation: 과거 기울기(gradient) 정보를 활용하여 각 매개변수에 대한 학습률을 적응적으로 조절합니다. 이를 통해 학습 속도를 높이고 최적의 모델 성능에 빠르게 도달하도록 돕습니다.
Momentum: 이전 기울기의 방향을 고려하여 학습 과정에 관성을 부여합니다. 이는 학습 과정에서 발생하는 진동을 줄이고 안정적인 수렴을 촉진합니다.
효율적인 학습: 희소한(sparse) 데이터 또는 노이즈가 많은 데이터에서도 안정적으로 학습할 수 있도록 돕습니다.

컴퓨터 비전 분야에서 LBFGS(Limited-memory Broyden–Fletcher–Goldfarb–Shanno) 알고리즘은 딥러닝 모델 학습에 사용되는 2차 미분 기반 최적화 알고리즘입니다.

LBFGS의 주요 특징:

2차 미분 정보 활용: 2차 미분 정보(Hessian)를 근사하여 학습 방향을 결정합니다. 이는 곡률 정보를 반영하여 학습 효율성을 높이고 최적점에 빠르게 도달하도록 돕습니다.
제한된 메모리 사용: 전체 Hessian 행렬을 저장하지 않고 과거 기울기 정보를 활용하여 Hessian을 근사합니다. 이를 통해 메모리 사용량을 줄이고 대규모 모델 학습에 적용할 수 있습니다.
'''

# Pytorch GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available() else 256

for epoch in range(1, 30):
    # 환경 변수
    style_img_path = 'panda_images/fubao/fubao_' + str(epoch) + '.png'
    content_img_path = 'images/bear/bear_1.jpg'
    output_model_path = 'saved_models/panda_style_model_adain.pth'

    # 이미지 로드
    style_img = image_loader(style_img_path, imsize).to(device)
    content_img = image_loader(content_img_path, imsize).to(device)

    # 사전 학습된 VGG19 모델 로드
    cnn = models.vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # model.py에서 정의한 모델과, Style, Content Loss 클래스 객체 생성
    model, style_losses, content_losses = get_model_and_losses(cnn, cnn_normalization_mean, cnn_normalization_std, style_img, content_img)

    # 입력 이미지 초기화
    input_img = content_img.clone()

    # Optimizer : AdaIN 방식은 Adam이 가장 적합하다
    # SparseAdam은 못씀, ASGD, SGD는 아예 별로
    optimizer = optim.Adam([input_img.requires_grad_()], lr=0.01) 

    # 하이퍼파라미터 설정
    num_steps = 300
    style_weight = 1e6 # 기본은 1e6
    content_weight = 1 # 기본은 1

    print('Optimizing...')
    for step in range(num_steps):
        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = sum([sl.loss for sl in style_losses])
            content_score = sum([cl.loss for cl in content_losses])

            loss = style_weight * style_score + content_weight * content_score
            loss.backward()
            return loss

        optimizer.step(closure)

        if step % 50 == 0:
            style_score = sum([sl.loss for sl in style_losses])
            content_score = sum([cl.loss for cl in content_losses])
            print(f"Step {step}:")
            print(f"Style Loss : {style_score.item():4f} Content Loss: {content_score.item():4f}")

    input_img.data.clamp_(0, 1)

    # 최종 이미지와 학습된 모델 저장
    output_image_path = 'output_images/AdaIN/panda_styled_image_' + str(epoch) + '.jpg'
    save_image(input_img, output_image_path)

    torch.save(model.state_dict(), output_model_path)
    print("Training completed and model saved.")



'''
# Pytorch GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available() else 256

# 환경 변수
style_img_path = 'panda_images/fubao/fubao_2.jpg'
content_img_path = 'images/bear/bear_2.jpg'
output_model_path = 'saved_models/panda_style_model_adain.pth'

# 이미지 로드
style_img = image_loader(style_img_path, imsize).to(device)
content_img = image_loader(content_img_path, imsize).to(device)

# 사전 학습된 VGG19 모델 로드
cnn = models.vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# model.py에서 정의한 모델과, Style, Content Loss 클래스 객체 생성
model, style_losses, content_losses = get_model_and_losses(cnn, cnn_normalization_mean, cnn_normalization_std, style_img, content_img)

# 입력 이미지 초기화
input_img = content_img.clone()

# Optimizer : AdaIN 방식은 Adam이 가장 적합하다
# SparseAdam은 못씀, ASGD, SGD는 아예 별로
optimizer = optim.Adam([input_img.requires_grad_()], lr=0.01) 

# 하이퍼파라미터 설정
num_steps = 300
style_weight = 1e6 # 기본은 1e5
content_weight = 1 # 기본은 1

print('Optimizing...')
for step in range(num_steps):
    def closure():
        input_img.data.clamp_(0, 1)

        optimizer.zero_grad()
        model(input_img)
        style_score = sum([sl.loss for sl in style_losses])
        content_score = sum([cl.loss for cl in content_losses])

        loss = style_weight * style_score + content_weight * content_score
        loss.backward()
        return loss

    optimizer.step(closure)

    if step % 50 == 0:
        style_score = sum([sl.loss for sl in style_losses])
        content_score = sum([cl.loss for cl in content_losses])
        print(f"Step {step}:")
        print(f"Style Loss : {style_score.item():4f} Content Loss: {content_score.item():4f}")

input_img.data.clamp_(0, 1)

# 최종 이미지와 학습된 모델 저장
save_image(input_img, 'output_images/panda_styled_image_adain.jpg')
torch.save(model.state_dict(), output_model_path)
print("Training completed and model saved.")
'''