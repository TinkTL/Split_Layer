from PIL import Image, ImageDraw
import numpy as np
from model import OCRProcessor
from skimage.measure import label, regionprops
import json
import time
import os
import requests
from io import BytesIO


def load_config():
    """加载配置文件"""
    with open('config.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def download_image(url):
    """
    从URL下载图片
    Args:
        url: 图片的URL地址
    Returns:
        PIL.Image对象
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # 检查请求是否成功
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"下载图片时出错: {e}")
        return None


def process_image_from_url(image_url, min_area=1000):
    """
    从URL处理图像的主函数
    Args:
        image_url: 图片的URL地址
        min_area: 最小区域面积
    """
    config = load_config()

    # 确保输出目录存在
    os.makedirs(config['output_path'], exist_ok=True)

    # 从URL下载图片
    input_image = download_image(image_url)
    if input_image is None:
        print("图片下载失败")
        return None

    # 处理图像
    processed = (extract_alpha_mask(input_image)
                if input_image.mode == 'RGBA'
                else input_image)
    masked_image = mask_text(processed)
    layers = separate_layers(masked_image, input_image, min_area)

    # 保存图层
    # 从URL中提取文件名，如果没有则使用时间戳
    image_name = os.path.basename(image_url.split('?')[0])  # 移除URL参数
    if not image_name:
        image_name = f"image_{int(time.time())}.png"

    save_layers(layers, image_name, config['output_path'])

    return layers


def extract_alpha_mask(image):
    """从RGBA图像中提取并反转alpha通道"""
    return Image.eval(image.split()[3], lambda x: 255 - x)


def mask_text(image):
    """识别并遮盖文字"""
    ocr = OCRProcessor()
    image_array = np.array(image)
    results = ocr.detect_text(image_array)

    result_image = image.convert('RGB')
    draw = ImageDraw.Draw(result_image)
    for bbox, _, _ in results:
        x0, y0 = bbox[0]  # 左上角
        x2, y2 = bbox[2]  # 右下角
        draw.rectangle([x0, y0, x2, y2], fill='black')

    return result_image


def separate_layers(mask_image, original_image, min_area=100):
    """将图像分离为不同的黑色区域图层"""
    gray_array = np.array(mask_image.convert("L"))
    mask = gray_array < 128
    regions = regionprops(label(mask))

    layers = []
    for region in regions:
        if region.area >= min_area:
            layer = create_layer(region, mask, label(mask), original_image)
            layers.append(layer)

    return layers


def create_layer(region, mask, label_image, original_image):
    """
    创建单个图层，只保留实际区域大小
    """
    # 获取区域边界框
    minr, minc, maxr, maxc = region.bbox

    # 计算区域的实际宽度和高度
    height = maxr - minr
    width = maxc - minc

    # 创建该区域大小的遮罩
    region_mask = mask[minr:maxr, minc:maxc] & (
        label_image[minr:maxr, minc:maxc] == region.label
    )

    # 创建裁剪后的透明图层
    layer_image = create_transparent_layer(region_mask, original_image, (minr, minc, maxr, maxc))

    return {
        'image': layer_image,
        'bbox': {
            'width': width,
            'height': height,
            'original_position': {
                'top': minr,
                'left': minc,
                'bottom': maxr,
                'right': maxc,
                'center': region.centroid
            }
        }
    }


def create_transparent_layer(mask, original_image, bbox):
    """
    创建透明背景的图层，只保留实际区域
    """
    minr, minc, maxr, maxc = bbox

    # 获取原图中对应区域的数据
    if original_image.mode == 'RGBA':
        original_array = np.array(original_image)[minr:maxr, minc:maxc]
    else:
        original_array = np.array(original_image)[minr:maxr, minc:maxc]

    # 创建对应大小的透明图层
    layer_image = np.zeros((maxr-minr, maxc-minc, 4), dtype=np.uint8)

    if original_image.mode == 'RGBA':
        layer_image[mask] = original_array[mask]
    else:
        rgb_data = original_array[mask]
        if len(rgb_data.shape) == 2:
            layer_image[mask] = np.column_stack((
                rgb_data, rgb_data, rgb_data,
                np.full(rgb_data.shape[0], 255)
            ))
        else:
            layer_image[mask] = np.column_stack((
                rgb_data,
                np.full(rgb_data.shape[0], 255)
            ))

    return layer_image


def save_layers(layers, image_name, output_path):
    """保存图层和位置信息"""
    base_name = os.path.splitext(image_name)[0]
    timestamp = int(time.time() * 1000)

    for i, layer_data in enumerate(layers, 1):
        filename = f"{timestamp}_{base_name}_layer_{i}.png"
        output_file = os.path.join(output_path, filename)
        Image.fromarray(layer_data['image'], 'RGBA').save(output_file)

        layer_info = {
            'width': layer_data['bbox']['width'],
            'height': layer_data['bbox']['height'],
            'original_position': layer_data['bbox']['original_position']
        }
        print_layer_info(filename, layer_info)


def print_layer_info(filename, layer_info):
    """打印图层信息"""
    print(f"\nlayer_name: {filename}")
    print(
        f"size: {layer_info['width']}x{layer_info['height']} px\n"
        f"locate: "
        f"({layer_info['original_position']['left']}, "
        f"{layer_info['original_position']['top']}), "
        f"({layer_info['original_position']['right']}, "
        f"{layer_info['original_position']['bottom']})"
    )


if __name__ == "__main__":
    # 从用户输入获取URL
    image_url = "https://image.lainuoniao.cn/sample_image.png"
    print(f"\n处理图片: {image_url}")
    process_image_from_url(image_url, min_area=100)