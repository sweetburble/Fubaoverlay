import os
from PIL import Image
import time

def resize_images(folder_path):


    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            try:
                with Image.open(image_path) as image:
                    # resized_image = image.resize((256, 256))
                    resized_image = image.convert("RGB")
                    resized_image.save(image_path)
                # 캐시 무효화를 위해 파일을 다시 열고 닫기
                with Image.open(image_path):
                    pass
                print(f"Processed {filename}")
                # 너무 많은 파일을 동시에 처리하지 않도록 잠시 대기
                time.sleep(0.01)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")


folder_path = './temp'
resize_images(folder_path)

