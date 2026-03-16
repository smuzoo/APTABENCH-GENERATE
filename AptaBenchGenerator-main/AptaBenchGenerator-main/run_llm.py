import sys
sys.path.append('aptamer_model')
from src.llm_generator import LLMGenerator
import pandas as pd
import os
from dotenv import load_dotenv

# Загрузка переменных из .env файла
load_dotenv()

# Получение API-ключа из переменной окружения
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY не найден в переменных окружения. Установите его в .env файле или переменной окружения.")

# Выбор prompt: 1 - старый, 2 - новый (с ограничениями)
prompt_choice = input("Выберите prompt: 1 (старый) или 2 (новый с ограничениями): ").strip()
if prompt_choice not in ['1', '2']:
    print("Неверный выбор, используем 1.")
    prompt_choice = '1'

# Инициализация генератора с выбором prompt
generator = LLMGenerator(api_key, prompt_version=int(prompt_choice))

# Генерация и оценка: 10 последовательностей в 3 итерациях
best_sequences = generator.generate_and_evaluate(num_sequences=10, iterations=3)

# Вывод результатов
print("Лучшие аптамеры (с вероятностью >0.5):")
for seq in best_sequences:
    print(seq)

# Сохранение ВСЕХ сгенерированных аптамеров (с probas и labels)
# Генерируем финальный батч для сохранения всех
all_sequences = generator.generate_sequences(50)  # Генерируем 50 для примера
all_results = generator.evaluate_sequences(all_sequences)  # Оцениваем все

if all_results:
    import os
    file_path = 'all_generated_aptamers_v2.csv' if prompt_choice == '2' else 'all_generated_aptamers.csv'
    file_exists = os.path.isfile(file_path)
    
    # Создаем DataFrame со всеми
    sequences, probas = zip(*all_results)
    labels = [1 if p > 0.5 else 0 for p in probas]  # Label на основе порога
    
    data = {
        'sequence': sequences,
        'canonical_smiles': [generator.target_smiles] * len(sequences),
        'proba': probas,
        'label': labels,
        'origin': ['LLM_generated'] * len(sequences),
        'source': ['Gemini_AptaBench_v2' if prompt_choice == '2' else 'Gemini_AptaBench'] * len(sequences)
    }
    df = pd.DataFrame(data)
    
    # Append или create
    df.to_csv(file_path, mode='a', header=not file_exists, index=False)
    print(f"Все результаты добавлены в {file_path} ({len(sequences)} записей, с разными labels)")
else:
    print("Нет сгенерированных аптамеров.")