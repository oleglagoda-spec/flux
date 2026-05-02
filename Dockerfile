# Используем образ с поддержкой CUDA и Python
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y git

# Устанавливаем библиотеки для генерации картинок и RunPod SDK
RUN pip install runpod diffusers transformers accelerate safetensors

# Копируем наш будущий обработчик
COPY handler.py /handler.py

# Команда запуска
CMD [ "python", "-u", "/handler.py" ]
