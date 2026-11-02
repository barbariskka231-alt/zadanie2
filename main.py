import streamlit as tf
import easyocr
import numpy as np
from PIL import Image
from pdf2image import convert_from_bytes
import re

# Настройка страницы веб-интерфейса
tf.set_page_config(page_title="ИИ Распознавание Договоров", layout="wide", page_icon="📄")

tf.title("📄 Интеллектуальное распознавание сканов договоров")
tf.write("Загрузите скан договора в формате изображения или PDF, чтобы извлечь текст и ключевые параметры.")

# Инициализация модели EasyOCR (кэшируем, чтобы не загружать при каждом действии)
@tf.cache_resource
def load_ocr_model():
    # Загружает детекторы текста и распознаватели для русского и английского языков
    return easyocr.Reader(['ru', 'en'], gpu=False)

reader = load_ocr_model()

def extract_contract_metadata(text):
    """
    Упрощенный алгоритм регулярных выражений / NLP для поиска сущностей.
    В продакшене этот блок заменяется на модель Transformers (например, RuBERT-NER).
    """
    metadata = {}
    
    # Поиск номера договора
    num_match = re.search(r'(?:договор|№)\s*(?:от)?\s*([№\d\s\-\/А-Яа-я]+)', text, re.IGNORECASE)
    metadata['Номер договора'] = num_match.group(1).strip() if num_match else "Не найден"
    
    # Поиск даты
    date_match = re.search(r'(\d{2}[\.\/]\d{2}[\.\/]\d{4}|\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+\d{4})', text, re.IGNORECASE)
    metadata['Дата заключения'] = date_match.group(1).strip() if date_match else "Не найдена"
    
    return metadata

# Панель загрузки файлов
uploaded_file = tf.file_uploader("Выберите файл скана", type=["png", "jpg", "jpeg", "pdf"])

if uploaded_file is not None:
    images = []
    
    # 1. Обработка загруженного файла
    if uploaded_file.type == "application/pdf":
        with tf.spinner("Конвертируем страницы PDF в изображения..."):
            pdf_bytes = uploaded_file.read()
            images = convert_from_bytes(pdf_bytes)
    else:
        image = Image.open(uploaded_file)
        images.append(image)
        
    tf.success(f"Успешно загружено страниц: {len(images)}")
    
    # Отображение превью первой страницы
    with tf.expander("👁 Показать превью первой страницы"):
        tf.image(images[0], use_container_width=True)
        
    # Кнопка запуска нейросети
    if tf.button("🚀 Запустить распознавание ИИ"):
        full_text = ""
        
        progress_bar = tf.progress(0)
        for idx, img in enumerate(images):
            tf.write(f"Обработка страницы {idx + 1}...")
            
            # Конвертируем PIL Image в numpy array для EasyOCR
            img_np = np.array(img)
            
            # 2. Распознавание текста нейросетью
            result = reader.readtext(img_np, detail=0)
            page_text = "\n".join(result)
            full_text += f"\n--- Страница {idx + 1} ---\n" + page_text
            
            progress_bar.progress((idx + 1) / len(images))
            
        tf.success("Распознавание успешно завершено!")
        
        # Разделение интерфейса на две колонки для вывода результатов
        col1, col2 = tf.columns(2)
        
        with col1:
            tf.subheader("📋 Извлеченные параметры договора")
            metadata = extract_contract_metadata(full_text)
            for key, val in metadata.items():
                tf.metric(label=key, value=val)
                
        with col2:
            tf.subheader("📝 Полный распознанный текст")
            tf.text_area("Текст из скана:", full_text, height=400)
            
            # Возможность скачать распознанный текст
            tf.download_button(
                label="📥 Скачать текст файлом",
                data=full_text,
                file_name="recognized_contract.txt",
                mime="text/plain"
            )
