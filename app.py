from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
import random
from apply_style import *  # Neural Style Transfer 적용

app = Flask(__name__)

# 이미지 저장 경로 설정
UPLOAD_FOLDER = 'images'
OUTPUT_FOLDER = 'output_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# 업로드된 이미지를 위한 라우트
@app.route('/uploads/<filename>')
def uploaded_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# 변환된 이미지를 위한 라우트
@app.route('/output/<filename>')
def output_image(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

# 이미지 업로드 및 변환 처리
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        # apply_style.py의 logic 함수 호출
        logic(filename)
        
        # 변환된 이미지 파일 경로 생성
        output_filename = f'[panda]{filename}'
        output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # 변환된 이미지 웹페이지에 표시
        return render_template('index.html', original_image=filename, panda_image=output_filename)


'''
/result 폴더에서 랜덤한 이미지 파일 선택
'''
# 이미지 파일 목록
image_files = os.listdir('result')
# 아직 선택되지 않은 이미지 파일 목록
unselected_files = image_files.copy()

@app.route('/random')
def random_image():
    return render_template('random.html')

@app.route('/result/<filename>')
def result_images(filename):
    return send_from_directory('result', filename)


@app.route('/get_random_image')
def get_random_image():
    global unselected_files
    
    # 모든 이미지 파일을 한 번씩 선택했다면 unselected_files 리셋
    if not unselected_files:
        unselected_files = image_files.copy()
    
    # 랜덤한 이미지 파일 선택
    selected_image = random.choice(unselected_files)
    unselected_files.remove(selected_image)

    return jsonify({'image': selected_image})


# 메인 페이지
@app.route('/')
def index():
    return render_template('index.html')

# dog2fubao 페이지
@app.route('/dog2fubao')
def dog2fubao():
    return render_template('dog2fubao.html')

# cat2fubao 페이지
@app.route('/cat2fubao')
def cat2fubao():
    return render_template('cat2fubao.html')

# bear2fubao 페이지
@app.route('/bear2fubao')
def bear2fubao():
    return render_template('bear2fubao.html')

# human2fubao 페이지
@app.route('/human2fubao')
def human2fubao():
    return render_template('human2fubao.html')

# 파일 이름을 안전하게 만든다
def secure_filename(filename):
    return filename.replace(' ', '_').lower()

if __name__ == '__main__':
    app.run(debug=True)