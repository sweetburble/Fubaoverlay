import torch
import torch.optim as optim
from torchvision import models
from torchvision.models import VGG19_Weights
from helpers import image_loader, save_image
from model import get_model_and_losses

'''
Neural Style Transfer 모델을 학습한다

'''

# Pytorch GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available() else 256 

# 환경 변수
style_img_path = 'panda_images/fubao/fubao_27.png'
content_img_path = 'images/bear/bear_1.jpg'
output_model_path = 'saved_models/panda_style_model.pth'

# 이미지 로드
style_img = image_loader(style_img_path, imsize).to(device)
content_img = image_loader(content_img_path, imsize).to(device)

# 사전 학습된 VGG19 모델 로드
cnn = models.vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# model.py에서 정의한 모델과, Style, Content Loss 클래스 객체 생성
model, style_losses, content_losses = get_model_and_losses(
    cnn, cnn_normalization_mean, cnn_normalization_std, style_img, content_img)

# 입력 이미지 초기화
input_img = content_img.clone()

# Optimizer : 순정은 LBFGS가 가장 적합하다
# SparseAdam은 못씀, ASGD, SGD는 아예 별로
optimizer = optim.LBFGS([input_img.requires_grad_()])

# 하이퍼파라미터 설정
num_steps = 300
style_weight = 1e6 # 기본은 1e6
content_weight = 1 # 기본은 1

print('Optimizing...')
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
save_image(input_img, 'output_images/panda_styled_image.jpg')
torch.save(model.state_dict(), output_model_path)
print("Training completed and model saved.")



'''
for epoch in range(2, 31):
    # 환경 변수
    style_img_path = 'panda_images/fubao/fubao_27.png'
    content_img_path = 'images/bear/bear_' + str(epoch) + '.jpg'
    output_model_path = 'saved_models/panda_style_model.pth'

    # 이미지 로드
    style_img = image_loader(style_img_path, imsize).to(device)
    content_img = image_loader(content_img_path, imsize).to(device)

    # 사전 학습된 VGG19 모델 로드
    cnn = models.vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # model.py에서 정의한 모델과, Style Loss, Content Loss 클래스의 객체 리스트 반환
    model, style_losses, content_losses = get_model_and_losses(
        cnn, cnn_normalization_mean, cnn_normalization_std, style_img, content_img)

    # 입력 이미지 초기화
    input_img = content_img.clone()

    # Optimizer : 순정은 LBFGS가 가장 적합하다
    # SparseAdam은 못씀, ASGD, SGD는 아예 별로
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    # 하이퍼파라미터 설정
    num_steps = 300
    style_weight = 1e6 # 기본은 1e6
    content_weight = 1 # 기본은 1

    print('Optimizing...')
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
    output_image_path = 'output_images/bear/panda_styled_image_' + str(epoch) + '.jpg'
    save_image(input_img, output_image_path)
    torch.save(model.state_dict(), output_model_path)
    print("Training completed and model saved.")
'''


'''

'''