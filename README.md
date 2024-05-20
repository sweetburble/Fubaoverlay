#### 푸바오가 너무 귀엽다

![fubao_19](https://github.com/sweetburble/Fubaoverlay/assets/79733289/31f14b16-c3f5-42d2-aa6a-6b752e70a6c6)
<br><br><br>

#### 귀여워

![fubao_7](https://github.com/sweetburble/Fubaoverlay/assets/79733289/15f7eb4e-6a79-43e3-b784-d79589abc6f6)
<br><br><br>

#### 너무 귀여워

![fubao_22](https://github.com/sweetburble/Fubaoverlay/assets/79733289/86223406-5a3a-4efb-8432-77ea91eb705b)
<br><br><br>

#### 푸바오.. 너는 내 삶에 희망이야

![fubao_25](https://github.com/sweetburble/Fubaoverlay/assets/79733289/436d8609-9642-4fdc-a4b2-12b6db72b3e7)
<br><br><br>

#### 그러던 어느날, 푸바오가 내 곁을 떠나버렸다

![fubao_return](https://github.com/sweetburble/Fubaoverlay/assets/79733289/febe48a0-d9ed-4f24-a032-e5ce05906da7)
<br><br><br>

### 안돼, 이럴수는 없어... 어떻게든 방법을 찾아야 해

.  
.  
.  
.  
.  
.

# 🐼 FubaOverlay : 세상 모든 것을 푸바오로

혹시 푸바오 좋아하세요? 세상 모든 것을 푸바오로 바꿀 수 있다면 어떨까요?

**FubaOverlay**는 여러분의 상상을 현실로 만들어 드립니다!

<br><br>

## 🚀 1. FubaOverlay, 넌 누구냐?

"FubaOverlay"는 **Neural Style Transfer**와 **CycleGAN**이라는 기술을 이용하여 이미지를 귀여운 판다 스타일로 변환해주는 웹사이트입니다. 이미지를 업로드하면, 순식간에 판다 이미지로 변신!

<br><br>

## 🎨 2. Neural Style Transfer vs CycleGAN : 마법의 비밀

### 🤖 Neural Style Transfer (NST)

이미지의 "콘텐츠"와 다른 이미지의 "스타일"을 결합하여, 새로운 이미지를 만드는 기술입니다.  
쉽게 말하자면, 한 이미지의 내용을 유지하면서 다른 이미지의 "푸바오"적인 스타일을 입히는 것이죠.

#### 핵심 원리

NST의 핵심은 **합성곱 신경망** (Convolutional Neural Network, CNN)을 사용하여 이미지의 콘텐츠와 스타일을 각각 표현하는 것입니다.

콘텐츠 표현 : CNN의 중간 레이어는 입력 이미지의 내용을 추상적인 형태로 표현합니다. 이러한 표현은 이미지의 객체, 구성, 공간적 배치 등을 담고 있습니다.  
스타일 표현 : CNN의 여러 레이어에서 추출된 Feature Map 간의 상관관계를 분석하여 이미지의 스타일을 표현합니다. 이는 색상, 질감, 붓터치, 패턴 등의 예술적인 요소를 나타냅니다.  
NST는 콘텐츠 이미지와 스타일 이미지의 표현을 합성하여 새로운 이미지를 생성합니다. 이 과정은 콘텐츠 이미지의 내용을 유지하면서 스타일 이미지의 스타일을 적용하는 "최적화 문제"로 해결됩니다.

#### NST 작동 방식

CNN 학습 : 먼저, 이미지 분류 등의 작업을 위해 미리 학습된 CNN 모델을 사용합니다. 이 프로젝트에서는 VGG19 모델을 사용했습니다.  
콘텐츠 및 스타일 표현 추출 : 콘텐츠 이미지와 스타일 이미지를 CNN에 입력하여 각각의 콘텐츠 및 스타일 표현을 추출합니다. Model.py에 구현되어 있습니다.  
최적화 : 초기 랜덤 이미지를 생성하고, CNN을 통해 콘텐츠 및 스타일 손실을 계산합니다. 이 손실을 최소화하는 방향으로 이미지를 반복적으로 업데이트하여 최종 합성 이미지를 생성합니다. train_style.py에 구현되어 있습니다.

<br><br>

### 🤖 CycleGAN

CycleGAN은 도메인 사이의 Image style transfer에 사용되는 Generative Adversarial Network(GAN) 입니다.  
이 작업은 비지도학습(unsupervised learning)으로 수행됩니다. 즉, 두 도메인 모두에서 이미지를 일대일 매핑 하지 않습니다.

### How do they work? (CycleGAN은 어떻게 동작하는 것일까?)

**CycleGAN 의 목표는, Pair-Image를 이루는 데이터가 없어도, 두 개의 도메인 X 와 Y 사이의 mapping function 을 학습하는 것입니다.**

![cyclegan_1](https://github.com/sweetburble/Fubaoverlay/assets/79733289/733cc8c2-44a4-493d-af72-2fa6447c3ed6)

CycleGAN의 핵심 아이디어는 두 개의 **생성자**(Generator)와 두 개의 **판별자**(Discriminator)를 사용하는 것입니다.

생성자(Generator) : 도메인 X의 이미지를 도메인 Y의 이미지로 변환하는 G_XtoY와, 그 반대인 G_YtoX 두 개의 생성자가 사용됩니다.

판별자(Discriminator) : 생성된 이미지가 진짜 도메인 Y (또는 X)에서 왔는지 판별하는 D_X와 D_Y 두 개의 판별자가 사용됩니다.

**Cycle-Consistency Loss** : G_XtoY를 다시 G_YtoX에 통과시켜 원래의 X와 유사한 이미지가 나오도록 합니다. 이를 통해 변환의 안정성을 높입니다.

**Adversarial Loss** : 생성자가 만든 가짜 이미지를 판별자가 진짜로 판별하도록 경쟁적으로 학습시킵니다.

이러한 구조를 통해 CycleGAN은 쌍을 이루는 데이터셋 없이도 안정적으로 이미지 변환을 수행할 수 있습니다.

<br><br>

## ✨ 3. 그래서 뭘 할 수 있냐고요? ✨

1. **메인 화면 :** 이미지를 업로드하고 "Neural Style Transfer" 버튼을 누르고 잠깐만 기다리면, 귀여운 푸바오 스타일로 변환된 이미지를 바로 확인할 수 있습니다.

2. **랜덤 페이지 :** 미리 만들어 둔 다양한 푸바오 변환 이미지들을 랜덤하게 감상하실 수 있습니다! 🖼️

3. **Dog2Fubao, Cat2Fubao, Bear2Fubao, Human2Fubao 페이지 :** NST보다 더욱 강력한 변환! 개, 고양이, 곰, 심지어 사람까지 판다로 변신시키는 마법을 경험해보세요. 🐶🐱🐻🧑

<br><br>

## 😀 4. 지금 바로 FubaOverlay를 경험해 보세요

FubaOverlay는 누구나 쉽고 재미있게 사용할 수 있습니다. 지금 바로 웹사이트를 실행하여 푸바오의 세계로 빠져보세요!

#### 주의!!! 변환 결과는 예상과 다를 수 있습니다. (하지만 늘 귀여울 거예요.. 푸바오처럼! 😉)

<br><br>

> **개발 및 실행 환경**  
> OS : Window 11  
> CPU : Ryzen 5 5600X  
> GPU : RTX 3070 8GB  
> RAM : 32GB  
> Python : 3.10.14  
> 브라우저 : Chrome

<br>

#### 1. requirements.txt에 있는 라이브러리들을 설치합니다

```bash
pip install -r requirements.txt
```

<br>

#### 2. app.py를 직접 실행하거나, 다음과 같은 명령어로 웹사이트를 실행합니다

그리고 웹 브라우저에서 <http://127.0.0.1:5000/> 에 접속하여 FubaOverlay를 경험할 수 있습니다.

```bash
python app.py
```

<br><br>

#### 3. 메인 페이지는 다음과 같습니다

![mainpage_1](https://github.com/sweetburble/Fubaoverlay/assets/79733289/2378931e-0eaa-4105-aceb-e12e62a53e3f)

각각 Random, Bear2Fubao, Dog2Fubao, Human2Fubao, Cat2Fubao 페이지로 이동하는 플로팅버튼이 존재합니다.

<br>

그 아래에는 NST 방식으로 이미지를 변환하기 위한 UI가 있습니다.
"파일 선택" 버튼을 눌러 이미지를 선택하고, "Upload" 버튼을 누르고 잠시만 기다리면 변환된 이미지를 확인할 수 있습니다.

![mainpage_2](https://github.com/sweetburble/Fubaoverlay/assets/79733289/95b13fa7-1566-458c-8e19-926381a7995e)

<br><br>

#### 4. Random 페이지는 다음과 같습니다

![random_1](https://github.com/sweetburble/Fubaoverlay/assets/79733289/f70f7a23-54fc-4fa1-8725-f021353c6ec2)
직접 CycleGAN으로 생성한 이미지들을 랜덤하게 감상할 수 있습니다.  
이미지를 클릭하면, 다음 이미지로 넘어갑니다.

<br><br>

#### 5. Bear2Fubao, Dog2Fubao, Human2Fubao, Cat2Fubao 페이지는 다음과 같습니다

![bear2fubao_1](https://github.com/sweetburble/Fubaoverlay/assets/79733289/529d2a4a-dd0f-4cfe-9c00-a325bd1d7935)
각각의 페이지에서는 해당 테마의 이미지를 푸바오로 변환한 결과를 잘 정리한 PDF로 확인할 수 있습니다.

**※ CycleGAN으로 작업한 소요시간은 다음과 같습니다.**

-   Bear2Fubao : 약 10시간
-   Dog2Fubao : 약 17시간
-   Human2Fubao : 약 15시간
-   Cat2Fubao : 약 17시간

<br><br><br>

## 🤝 5. 마치며 🤝

FubaoOverlay는 오픈 소스 프로젝트입니다. 당신의 소중한 의견을 기다릴게요!

**지금 바로 FubaoOverlay를 방문하고 세상을 푸바오로 물들여 보세요! 함께 즐거운 판다 세상을 만들어요!**  
아니면.. 루이바오, 후이바오도 괜찮아요 🐼🐼

<br><br><br>

## 📝 6. 참고자료 / Reference

[CycleGAN이란?]
<https://junyanz.github.io/CycleGAN/>

[PyTorch CycleGAN 구현]
<https://github.com/hanyoseob/pytorch-CycleGAN>

[CycleGAN 리뷰]
<https://mz-moonzoo.tistory.com/18>

[PyTorch로 Neural Style Transfer 구현하고 학습하기]
<https://deep-learning-study.tistory.com/680>

[AdaIN을 제대로 이해해보자]
<https://lifeignite.tistory.com/48>

[AdaIN 논문 리뷰]
<https://kyujinpy.tistory.com/65>  
**※ model_adain.py, train_style_adain.py, apply_style_adain.py 에 AdaIN을 적용한 Neural Style Transfer을 구현했지만, 최종 결과가 더 좋지 않아 소개하지 않았습니다.**

[코딩 및 에러 수정 도우미]
<https://chat.lmsys.org/>
