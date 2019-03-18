# YOLO-V3-IOU
YOLO3 动漫人脸检测 2019-1-19
## 数据集的准备 
- 数据集标签制作工具下载：https://github.com/tzutalin/labelImg
- 运行prepare.py将数据集划为训练集，验证集和测试集
## 训练模型
- 加载权重，将权重h5文件放入models文件夹
- 最新权重下载地址：https://pan.baidu.com/s/1QKODAhRU_Rw5lpf8NcIR4g
- 若重新开始训练，将load_pretrained置为False
- 设置学习率，若重新训练，建议设0.001，若加载权重，建议设较小的学习率
- 运行train.py
## 预测
- 加载权重，将训练好的权重h5文件放入models文件夹
- 运行run.py，对数据集进行预测输出，输出在outputs文件夹中
- 预测新的图片 
1. from predict import YOLO
2. from PIL import Image
3. yolo = YOLO()
4. img = Image.open(r'./test.jpg')
5. output = yolo.detect_image(img)
6. output.show()
## Demo
![](./outputs/Anime_311.jpg)
![](./outputs/Anime_250.jpg)
![](./outputs/Anime_305.jpg)
![](./outputs/Anime_289.jpg)
![](./outputs/Anime_301.jpg)
![](./outputs/Anime_258.jpg)
## 模型
![](./images/model.png)
