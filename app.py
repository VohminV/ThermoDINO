#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OFFLINE VISION INFERENCE SYSTEM - CPU OPTIMIZED
Автономная система визуального анализа
Оптимизирована для работы на CPU без потери функциональности
"""

import os
import uuid
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
import timm

# =========================================================
# PATH CONFIGURATION (ABSOLUTE PATHS)
# =========================================================

# Получаем абсолютный путь к директории со скриптом
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Определяем абсолютные пути к папкам
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, 'uploads')
RESULT_FOLDER = os.path.join(STATIC_FOLDER, 'results')
THERMAL_FOLDER = os.path.join(STATIC_FOLDER, 'thermal')
TEMPLATE_FOLDER = os.path.join(BASE_DIR, 'templates')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

# Гарантированное создание всех директорий при инициализации
for folder in [STATIC_FOLDER, UPLOAD_FOLDER, RESULT_FOLDER, THERMAL_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Проверка наличия папки шаблонов
if not os.path.exists(TEMPLATE_FOLDER):
    os.makedirs(TEMPLATE_FOLDER, exist_ok=True)
    print(f"[ПРЕДУПРЕЖДЕНИЕ] Папка шаблонов создана заново: {TEMPLATE_FOLDER}")

# =========================================================
# FLASK INITIALIZATION
# =========================================================

app = Flask(
    __name__, 
    static_folder=STATIC_FOLDER, 
    static_url_path='/static',
    template_folder=TEMPLATE_FOLDER
)

app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    RESULT_FOLDER=RESULT_FOLDER,
    THERMAL_FOLDER=THERMAL_FOLDER,
    MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH,
    SECRET_KEY=os.urandom(24)
)

print(f"[СИСТЕМА] Базовая директория: {BASE_DIR}")
print(f"[СИСТЕМА] Статическая папка: {STATIC_FOLDER}")
print(f"[СИСТЕМА] Папка загрузок: {UPLOAD_FOLDER}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dino_model = None

# =========================================================
# THERMAL ANALYSIS MODULE
# =========================================================

class ThermalAnalyzer:
    def __init__(self, temp_threshold=0.6):
        self.temp_threshold = temp_threshold
    
    def create_thermal_mask(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Не удалось прочитать изображение: {image_path}")
        
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(float) / 255.0
        thermal_data = l_channel
        
        thermal_min, thermal_max = thermal_data.min(), thermal_data.max()
        if thermal_max - thermal_min > 0:
            thermal_normalized = (thermal_data - thermal_min) / (thermal_max - thermal_min)
        else:
            thermal_normalized = thermal_data
        
        binary_mask = (thermal_normalized > self.temp_threshold).astype(np.uint8) * 255
        thermal_colored = cv2.applyColorMap((thermal_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        return thermal_normalized, binary_mask, thermal_colored
    
    def compare_thermal_signatures(self, mask_a, mask_b):
        intersection = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        iou = intersection / union if union > 0 else 0.0
        return round(iou * 100, 2)
    
    def compare_heat_distribution(self, heat_a, heat_b):
        if heat_a.shape != heat_b.shape:
            heat_b = cv2.resize(heat_b, (heat_a.shape[1], heat_a.shape[0]), interpolation=cv2.INTER_LINEAR)
            
        hist_a = cv2.calcHist([heat_a], [0], None, [256], [0, 256])
        hist_b = cv2.calcHist([heat_b], [0], None, [256], [0, 256])
        cv2.normalize(hist_a, hist_a, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_b, hist_b, 0, 1, cv2.NORM_MINMAX)
        correlation = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
        return round(max(0, correlation * 100), 2)
    
    def save_thermal_visualization(self, original_path, thermal_mask, output_filename):
        img = cv2.imread(original_path)
        if img is None:
            return None
        
        if thermal_mask.shape[:2] != img.shape[:2]:
            thermal_mask = cv2.resize(thermal_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        overlay = img.copy()
        mask_colored = np.zeros_like(img)
        mask_colored[thermal_mask > 0] = [0, 0, 255]
        cv2.addWeighted(mask_colored, 0.5, overlay, 0.5, 0, overlay)
        
        output_path = os.path.join(app.config['THERMAL_FOLDER'], output_filename)
        
        # Проверка успешности записи
        success = cv2.imwrite(output_path, overlay)
        if success:
            print(f"[СИСТЕМА] Визуализация сохранена: {output_path}")
            return url_for('static', filename=f'thermal/{output_filename}')
        else:
            print(f"[ОШИБКА] Не удалось сохранить визуализацию: {output_path}")
            return None

thermal_analyzer = ThermalAnalyzer(temp_threshold=0.6)

# =========================================================
# MODEL LOADING (CPU-OPTIMIZED)
# =========================================================

def load_dino_model():
    global dino_model
    if dino_model is None:
        print(f"[СИСТЕМА] Инициализация на устройстве: {device}")
        try:
            dino_model = timm.create_model(
                'vit_small_patch14_dinov2.lvd142m',
                pretrained=False,
                num_classes=0
            )
            
            model_path = os.path.join(BASE_DIR, "model.pth")
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=device, weights_only=True)
                model_dict = dino_model.state_dict()
                filtered = {k: v for k, v in state_dict.items() if k in model_dict}
                
                removed = set(state_dict.keys()) - set(filtered.keys())
                if removed:
                    print(f"[СИСТЕМА] Пропущено ключей: {len(removed)}")
                
                dino_model.load_state_dict(filtered, strict=True)
                print(f"[СИСТЕМА] Загружено {len(filtered)} параметров")
            
            dino_model = dino_model.to(device)
            dino_model.eval()
            print("[СИСТЕМА] Модель готова")
            
        except Exception as e:
            print(f"[ОШИБКА] {e}")
            raise
    return dino_model

# =========================================================
# INFERENCE (OPTIMIZED)
# =========================================================

def preprocess_image_for_dino(image_path, image_size=518):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return transform(img).unsqueeze(0).to(device)

def extract_dino_features(image_path):
    model = load_dino_model()
    img_tensor = preprocess_image_for_dino(image_path)
    
    with torch.no_grad():
        features = model(img_tensor)
    
    return F.normalize(features, p=2, dim=1).cpu().numpy()

def calculate_dino_similarity(img_a_path, img_b_path):
    features_a = extract_dino_features(img_a_path)
    features_b = extract_dino_features(img_b_path)
    similarity = cosine_similarity(features_a, features_b)[0][0]
    return round(similarity * 100, 2)

def calculate_ssim_opencv(img_a_path, img_b_path):
    img_a = cv2.imread(img_a_path, cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imread(img_b_path, cv2.IMREAD_GRAYSCALE)
    if img_a is None or img_b is None:
        raise ValueError("Ошибка чтения изображений")
    
    img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
    mse = np.mean((img_a.astype(float) - img_b.astype(float)) ** 2)
    if mse == 0:
        return 100.0
    return round(max(0, 100 * (1 - mse / (255.0 ** 2))), 2)

def calculate_thermal_similarity(img_a_path, img_b_path, filename_a, filename_b):
    heat_a, mask_a_orig, _ = thermal_analyzer.create_thermal_mask(img_a_path)
    heat_b, mask_b_orig, _ = thermal_analyzer.create_thermal_mask(img_b_path)
    
    mask_a_calc = mask_a_orig.copy()
    mask_b_calc = mask_b_orig.copy()
    heat_a_calc = heat_a.copy()
    heat_b_calc = heat_b.copy()
    
    if mask_a_calc.shape != mask_b_calc.shape:
        mask_b_calc = cv2.resize(mask_b_calc, (mask_a_calc.shape[1], mask_a_calc.shape[0]), interpolation=cv2.INTER_NEAREST)
        heat_b_calc = cv2.resize(heat_b_calc, (heat_a_calc.shape[1], heat_a_calc.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    iou_score = thermal_analyzer.compare_thermal_signatures(mask_a_calc > 0, mask_b_calc > 0)
    dist_score = thermal_analyzer.compare_heat_distribution(
        (heat_a_calc * 255).astype(np.uint8),
        (heat_b_calc * 255).astype(np.uint8)
    )
    
    thermal_viz_a = thermal_analyzer.save_thermal_visualization(
        img_a_path, mask_a_orig, f"thermal_{filename_a}"
    )
    thermal_viz_b = thermal_analyzer.save_thermal_visualization(
        img_b_path, mask_b_orig, f"thermal_{filename_b}"
    )
    
    autopilot_score = round((iou_score * 0.7) + (dist_score * 0.3), 2)
    
    return {
        'iou': iou_score,
        'distribution': dist_score,
        'autopilot_score': autopilot_score,
        'thermal_viz_a': thermal_viz_a,
        'thermal_viz_b': thermal_viz_b
    }

# =========================================================
# FLASK ROUTES
# =========================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    result_data = None
    error_message = None

    if request.method == 'POST':
        if 'image_a' not in request.files or 'image_b' not in request.files:
            error_message = "Требуется загрузка двух файлов."
            return render_template('index.html', error=error_message, result=None)

        file_a = request.files['image_a']
        file_b = request.files['image_b']

        if file_a.filename == '' or file_b.filename == '':
            error_message = "Файлы не выбраны."
            return render_template('index.html', error=error_message, result=None)

        if file_a and allowed_file(file_a.filename) and file_b and allowed_file(file_b.filename):
            try:
                # Гарантированное создание директории перед сохранением
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                
                filename_a = secure_filename(f"{uuid.uuid4().hex}_{file_a.filename}")
                filename_b = secure_filename(f"{uuid.uuid4().hex}_{file_b.filename}")
                
                path_a = os.path.join(app.config['UPLOAD_FOLDER'], filename_a)
                path_b = os.path.join(app.config['UPLOAD_FOLDER'], filename_b)
                
                print(f"[СИСТЕМА] Сохранение файла A: {path_a}")
                print(f"[СИСТЕМА] Сохранение файла B: {path_b}")
                
                file_a.save(path_a)
                file_b.save(path_b)
                
                # Проверка существования файлов после сохранения
                if not os.path.exists(path_a) or not os.path.exists(path_b):
                    raise FileNotFoundError("Файлы не были сохранены на диск")

                similarity_dino = calculate_dino_similarity(path_a, path_b)
                similarity_ssim = calculate_ssim_opencv(path_a, path_b)
                thermal_data = calculate_thermal_similarity(path_a, path_b, filename_a, filename_b)

                result_data = {
                    'similarity_dino': similarity_dino,
                    'similarity_ssim': similarity_ssim,
                    'thermal_iou': thermal_data['iou'],
                    'thermal_distribution': thermal_data['distribution'],
                    'thermal_autopilot_score': thermal_data['autopilot_score'],
                    'img_a': url_for('static', filename=f'uploads/{filename_a}'),
                    'img_b': url_for('static', filename=f'uploads/{filename_b}'),
                    'thermal_viz_a': thermal_data['thermal_viz_a'],
                    'thermal_viz_b': thermal_data['thermal_viz_b'],
                    'device': str(device)
                }

            except Exception as e:
                error_message = f"Ошибка обработки: {str(e)}"
                import traceback
                traceback.print_exc()
        else:
            error_message = "Недопустимый формат файла."

    return render_template('index.html', result=result_data, error=error_message)

if __name__ == '__main__':
    print("="*50)
    print("OFFLINE VISION SYSTEM - CPU OPTIMIZED")
    print(f"Device: {device}")
    print(f"Base Dir: {BASE_DIR}")
    print(f"Static Folder: {STATIC_FOLDER}")
    print("Features: DINOv2 + Thermal + Autopilot + Static")
    print("="*50)
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)