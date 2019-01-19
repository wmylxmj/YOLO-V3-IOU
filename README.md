# YOLO-V3-IOU
YOLO3 动漫人脸识别 2019-1-19
## 数据集的准备 
- 数据集标签制作工具下载：https://github.com/tzutalin/labelImg
- 运行prepare.py将数据集划为训练集，验证集和测试集
## 训练模型
- 加载权重，将权重h5文件放入models文件夹
- 若重新开始训练，将load_pretrained置为False
## 预测
- 运行run.py，对数据集进行预测输出，输出在outputs文件夹中
- 预测新的图片 
1. from predict import YOLO
2. from PIL import Image
3. yolo = YOLO()
4. img = Image.open(r'./test.jpg')
5. output = yolo.detect_image(img)
6. output.show()
