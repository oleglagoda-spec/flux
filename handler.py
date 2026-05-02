import runpod
import torch
from diffusers import FluxPipeline
import base64
from io import BytesIO

# 1. Инициализация модели
# Мы используем Flux.1-schnell, так как она быстрее и требует меньше шагов (4 шага)
def load_model():
    print("Загрузка модели Flux...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", 
        torch_dtype=torch.bfloat16
    ).to("cuda")
    return pipe

# Загружаем модель один раз при старте контейнера
try:
    pipe = load_model()
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")

# 2. Функция-обработчик запросов
def handler(job):
    job_input = job["input"]
    
    # Получаем промпт из запроса, если его нет — используем стандартный
    prompt = job_input.get("prompt", "A beautiful cinematic landscape")
    
    # Параметры генерации
    # Для версии Schnell guidance_scale должен быть 0.0, а шагов всего 4
    try:
        print(f"Генерация по промпту: {prompt}")
        image = pipe(
            prompt, 
            guidance_scale=0.0, 
            num_inference_steps=4, 
            max_sequence_length=256
        ).images[0]
        
        # Конвертируем готовую картинку в строку base64 для передачи по API
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {"image": img_str}
        
    except Exception as e:
        return {"error": str(e)}

# 3. Запуск воркера RunPod
runpod.serverless.start({"handler": handler})
