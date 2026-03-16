import sys
sys.path.append('aptamer_model')
from src.llm_generator import LLMGenerator
import pandas as pd

# Ваш API-ключ
api_key = "AIzaSyDiV5pHf-3tXrYMH8edV8DQxtNUkYmz8lE"

# Инициализация генератора
generator = LLMGenerator(api_key)

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
    file_path = 'all_generated_aptamers.csv'
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
        'source': ['Gemini_AptaBench'] * len(sequences)
    }
    df = pd.DataFrame(data)
    
    # Append или create
    df.to_csv(file_path, mode='a', header=not file_exists, index=False)
    print(f"Все результаты добавлены в {file_path} ({len(sequences)} записей, с разными labels)")
else:
    print("Нет сгенерированных аптамеров.")