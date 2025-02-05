from PIL import Image, ImageDraw
import numpy as np
from model import OCRProcessor
from skimage.measure import label, regionprops
import json
import time
import os


def load_config():
    """加载配置文件"""
    with open('config.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def process_image(input_image_path, min_area=1000):
    """主要的图像处理函数"""
    config = load_config()

    # 确保输出目录存在
    os.makedirs(config['output_path'], exist_ok=True)
    os.makedirs(config['output_json'], exist_ok=True)

    # 读取输入图像
    input_image = Image.open(input_image_path)

    # 处理图像
    processed = extract_alpha_mask(input_image) if input_image.mode == 'RGBA' else input_image
    masked_image = mask_text(processed)
    layers = separate_layers(masked_image, input_image, min_area)

    # 保存图层和JSON
    image_name = os.path.basename(input_image_path)
    save_layers(layers, image_name, config['output_path'], config['output_json'])

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
    """创建单个图层"""
    minr, minc, maxr, maxc = region.bbox

    layer_mask = np.zeros_like(mask, dtype=bool)
    layer_mask[minr:maxr, minc:maxc] = mask[minr:maxr, minc:maxc] & (
        label_image[minr:maxr, minc:maxc] == region.label
    )

    layer_image = create_transparent_layer(layer_mask, original_image)

    return {
        'image': layer_image,
        'bbox': {
            'top': minr,
            'left': minc,
            'bottom': maxr,
            'right': maxc,
            'center': region.centroid
        }
    }


def create_transparent_layer(mask, original_image):
    """创建透明背景的图层"""
    original_array = np.array(original_image)
    layer_image = np.zeros((*original_array.shape[:2], 4), dtype=np.uint8)

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


def save_layers(layers, image_name, output_path, output_json):
    """保存图层和位置信息"""
    base_name = os.path.splitext(image_name)[0]
    timestamp = int(time.time() * 1000)

    layers_info = {'total_layers': len(layers), 'layers': []}

    for i, layer_data in enumerate(layers, 1):
        # 保存图层图像
        filename = f"{timestamp}_{base_name}_layer_{i}.png"
        output_file = os.path.join(output_path, filename)
        Image.fromarray(layer_data['image'], 'RGBA').save(output_file)

        # 记录位置信息
        layer_info = {
            'layer_id': i,
            'file_path': output_file,
            'position': {
                'top_left': [
                    int(layer_data['bbox']['left']),
                    int(layer_data['bbox']['top'])
                ],
                'bottom_right': [
                    int(layer_data['bbox']['right']),
                    int(layer_data['bbox']['bottom'])
                ],
                'center': [
                    int(layer_data['bbox']['center'][1]),
                    int(layer_data['bbox']['center'][0])
                ]
            }
        }
        layers_info['layers'].append(layer_info)
        print_layer_info(filename, layer_info)

    # 保存JSON信息
    json_path = os.path.join(output_json, f"{base_name}_layers_info.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(layers_info, f, ensure_ascii=False, indent=2)


def print_layer_info(filename, layer_info):
    """打印图层信息"""
    print(f"\n已保存图层: {filename}")
    print(
        f"位置信息: "
        f"左上角({layer_info['position']['top_left'][0]}, "
        f"{layer_info['position']['top_left'][1]}), "
        f"右下角({layer_info['position']['bottom_right'][0]}, "
        f"{layer_info['position']['bottom_right'][1]}), "
        f"中心点({layer_info['position']['center'][0]}, "
        f"{layer_info['position']['center'][1]})"
    )


if __name__ == "__main__":
    # 示例使用
    config = load_config()
    input_image_path = os.path.join(config['input_path'], "sample_image.png")
    process_image(input_image_path, min_area=100)