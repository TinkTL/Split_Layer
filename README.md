# Split Layer

一个用于图像分层处理的 Python 工具。

## 功能
- 支持 RGBA 图像处理
- 文字识别与遮罩
- 图层分离
- JSON 输出位置信息

## 安装
1. 克隆仓库
2. 安装依赖：`pip install -r requirements.txt`

## 使用方法
1. 将待处理图片放入 input 文件夹
2. 运行 `python mian.py`
3. 处理结果将保存在 output 文件夹中

## 配置
在 config.json 中设置：
- input_path: 输入图片路径
- output_path: 输出图片路径
- output_json: JSON 输出路径