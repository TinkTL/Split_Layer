import os
import torch
import cv2
import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging

logger = logging.getLogger("EasyOCR")

model_dir_name = "EasyOCR"

lang_list = {
    "English": "en",
    "简体中文": "ch_sim",
    "繁體中文": "ch_tra",
    "العربية": "ar",
    "Euskal": "eu",
    "Bosanski": "bs",
    "Български": "bg",
    "Català": "ca",
    "Hrvatski": "hr",
    "Čeština": "cs",
    "Dansk": "da",
    "Nederlands": "nl",
    "Eesti": "et",
    "Suomi": "fi",
    "Français": "fr",
    "Galego": "gl",
    "Deutsch": "de",
    "Ελληνικά": "el",
    "עברית": "he",
    "हिन्दी": "hi",
    "Magyar": "hu",
    "Íslenska": "is",
    "Indonesia": "id",
    "Italiano": "it",
    "日本語": "ja",
    "한국어": "ko",
    "Latviešu": "lv",
    "Lietuvių": "lt",
    "Македонски": "mk",
    "Norsk": "no",
    "Polski": "pl",
    "Português": "pt",
    "Română": "ro",
    "Русский": "ru",
    "Српски": "sr",
    "Slovenčina": "sk",
    "Slovenščina": "sl",
    "Español": "es",
    "Svenska": "sv",
    "ไทย": "th",
    "Türkçe": "tr",
    "Українська": "uk",
    "Tiếng Việt": "vi",
}

def get_classes2(labels):
    # 如果输入是列表，直接处理
    if isinstance(labels, list):
        result = []
        for label in labels:
            if isinstance(label, str):
                for key, value in lang_list.items():
                    if label == key:
                        result.append(value)
                        break
        return result

    # 如果输入是字符串，按原来的方式处理
    label = labels.lower()
    labels = label.split(",")
    result = []
    for l in labels:
        for key, value in lang_list.items():
            if l == key:
                result.append(value)
                break
    return result

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    result = tgt["result"]

    res_mask = []
    res_image = []

    box_color = (255, 0, 0)
    text_color = (255, 255, 255)

    draw = ImageDraw.Draw(image_pil)
    font_size = 20

    try:
        font = ImageFont.truetype("docs/PingFang Regular.ttf", font_size)
    except OSError:
        logger.warning(f"Could not load font, using default font")
        font = ImageFont.load_default()

    labelme_data = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [],
        "imageHeight": H,
        "imageWidth": W,
    }

    for item in result:
        formatted_points, label, threshold = item
        x1, y1 = formatted_points[0]
        x2, y2 = formatted_points[2]
        threshold = round(threshold, 2)

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        points = [[x1, y1], [x2, y2]]

        shape = {
            "label": label,
            "points": points,
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {},
        }
        labelme_data["shapes"].append(shape)

        label = label + ":" + str(threshold)
        shape["threshold"] = str(threshold)

        draw.rectangle([(x1, y1), (x2, y2)], outline=box_color, width=3)

        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        draw.rectangle(
            [(x1, y1 - text_height - 10), (x1 + text_width, y1)], fill=box_color
        )
        draw.text((x1, y1 - text_height - 10), label, font=font, fill=text_color)

        mask = np.zeros((H, W, 1), dtype=np.uint8)
        cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), -1)
        mask_tensor = torch.from_numpy(mask).permute(2, 0, 1).float() / 255.0
        res_mask.append(mask_tensor)

    if len(res_mask) == 0:
        mask = np.zeros((H, W, 1), dtype=np.uint8)
        mask_tensor = torch.from_numpy(mask).permute(2, 0, 1).float() / 255.0
        res_mask.append(mask_tensor)

    image_with_boxes = np.array(image_pil)
    image_with_boxes_tensor = torch.from_numpy(
        image_with_boxes.astype(np.float32) / 255.0
    )
    image_with_boxes_tensor = torch.unsqueeze(image_with_boxes_tensor, 0)
    res_image.append(image_with_boxes_tensor)

    return res_image, res_mask, labelme_data


class OCRProcessor:
    """文字识别处理器"""
    def __init__(self):
        self.model_dir = "assets/models/EasyOCR"
        self._ensure_model_dir()

    def _ensure_model_dir(self):
        """确保模型目录存在"""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def detect_text(self, image, language="ch_sim"):
        """检测图像中的文字位置"""
        reader = easyocr.Reader(
            [language],
            gpu=torch.cuda.is_available(),
            model_storage_directory=self.model_dir
        )
        return reader.readtext(image)


# 支持的语言映射
LANGUAGE_CODES = {
    "简体中文": "ch_sim",
    "English": "en",
    "繁體中文": "ch_tra",
    "العربية": "ar",
    "বাংলা": "bn",
    "Български": "bg",
    "Català": "ca",
    "Čeština": "cs",
    "Dansk": "da",
    "Deutsch": "de",
    "Ελληνικά": "el",
    "Español": "es",
    "Eesti": "et",
    "فارسی": "fa",
    "Suomi": "fi",
    "Français": "fr",
    "עברית": "he",
    "हिन्दी": "hi",
    "Hrvatski": "hr",
    "Magyar": "hu",
    "Indonesia": "id",
    "Italiano": "it",
    "日本語": "ja",
    "ಕನ್ನಡ": "kn",
    "한국어": "ko",
    "Lietuvių": "lt",
    "Latviešu": "lv",
    "മലയാളം": "ml",
    "मराठी": "mr",
    "Melayu": "ms",
    "Nederlands": "nl",
    "Norsk": "no",
    "Polski": "pl",
    "Português": "pt",
    "Română": "ro",
    "Русский": "ru",
    "Slovenčina": "sk",
    "Slovenščina": "sl",
    "Српски": "sr",
    "Svenska": "sv",
    "தமிழ்": "ta",
    "తెలుగు": "te",
    "ไทย": "th",
    "Türkçe": "tr",
    "Українська": "uk",
    "اردو": "ur",
    "Tiếng Việt": "vi"
}

class ApplyEasyOCR:
    def main(self, image, gpu=False, mode="choose", language_list=None, language_name=None):
        #OCR支持批量，所以做了判断输入是单张图片还是列表
        if isinstance(image, Image.Image):
            image = np.array(image)

        if not isinstance(image, list):
            image = [image]

        results = []

        for item in image:
            # 确保图像是numpy数组
            if isinstance(item, Image.Image):
                item = np.array(item)

            image_pil = Image.fromarray(item)

            # 设置语言
            if mode == "choose" and language_list:
                language = get_classes2(language_list)
            else:
                language = [language_name]

            # 创建模型存储目录
            model_storage_directory = "assets/models/EasyOCR"
            if not os.path.exists(model_storage_directory):
                os.makedirs(model_storage_directory)

            # 创建reader并执行OCR
            reader = easyocr.Reader(language, gpu=gpu, model_storage_directory=model_storage_directory)
            ocr_result = reader.readtext(np.array(image_pil))

            # 返回识别结果
            results.extend(ocr_result)

        return results
