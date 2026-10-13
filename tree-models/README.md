# Tree Models — Decision Trees and Random Forests

Практический проект по деревьям решений и случайным лесам на `scikit-learn`.

Проект реализует упражнения из задания:

1. Decision Tree Regressor на California Housing.
2. Влияние `max_depth` и демонстрация bias-variance trade-off.
3. Random Forest Regressor и сравнение с Linear Regression и Decision Tree.
4. Feature importance Random Forest.
5. Decision Tree Classifier на Breast Cancer dataset.
6. Random Forest Classifier, classification report и top-5 признаков.

## Требования

- Python 3.10+
- `scikit-learn==1.7.2`

## Установка

Создай и активируй виртуальное окружение:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Установи зависимости:

```bash
pip install -r requirements.txt
```

## Запуск

```bash
python tree_models.py
```

Скрипт автоматически:

- скачивает California Housing dataset;
- заполняет пропуски в `total_bedrooms` медианой;
- обучает регрессионные деревья;
- исследует `max_depth` от 2 до 20;
- обучает Random Forest с 100 деревьями;
- считает feature importance;
- обучает Decision Tree и Random Forest на Breast Cancer dataset;
- печатает метрики и classification report;
- создаёт три изображения.

## Результаты

Ожидаемые результаты из задания могут немного отличаться из-за версий библиотек и численных особенностей.

### Decision Tree Regressor

Default tree:

- Train R² ≈ `1.0000`
- Test R² ≈ `0.6275`
- Depth ≈ `33`

Это показывает сильное переобучение: дерево практически запоминает обучающую выборку.

### `max_depth`

В задании лучший результат получается примерно при:

- `max_depth = 10`
- Test R² ≈ `0.6997`

После этого тестовый R² начинает снижаться, хотя train R² продолжает расти. Это классический bias-variance trade-off.

### Random Forest Regressor

Для 100 деревьев:

- Train R² ≈ `0.9746`
- Test R² ≈ `0.8101`

Ожидаемое сравнение:

| Model | Test R² |
|---|---:|
| LinearRegression | 0.6139 |
| DecisionTree (default) | 0.6275 |
| RandomForest (100) | 0.8101 |

Увеличение количества деревьев даёт быстрое улучшение в начале, после чего эффект постепенно насыщается.

### Feature importance

Самый важный признак:

`median_income`

Также среди наиболее важных:

- `longitude`
- `latitude`
- `housing_median_age`
- `population`

В задании top-5 признаков Random Forest совпадает с пятью признаками, оставшимися после Lasso.

### Breast Cancer

Decision Tree с `max_depth=4` показывает хорошую классификацию при сохранении интерпретируемости.

Random Forest с 100 деревьями также даёт высокую accuracy. В задании ожидается test accuracy около `0.9561`.

## Генерируемые файлы

После запуска появятся:

```text
ex02_depth.png
ex04_importance.png
ex05_tree.png
```

### `ex02_depth.png`

График Train/Test R² в зависимости от `max_depth`.

### `ex04_importance.png`

Горизонтальный bar chart важности признаков Random Forest.

### `ex05_tree.png`

Визуализация Decision Tree Classifier глубиной до 4 уровней.

## Структура проекта

```text
tree_models_project/
├── tree_models.py
├── requirements.txt
├── README.md
├── ex02_depth.png       # появляется после запуска
├── ex04_importance.png  # появляется после запуска
└── ex05_tree.png        # появляется после запуска
```

## Что демонстрирует проект

Проект показывает несколько важных идей машинного обучения:

- переобучение Decision Tree;
- влияние ограничения сложности модели через `max_depth`;
- bias-variance trade-off;
- снижение variance за счёт Random Forest;
- сравнение разных семейств моделей;
- интерпретацию `feature_importances_`;
- классификацию с Decision Tree;
- классификацию с Random Forest;
- использование `classification_report`.

## Источники

- scikit-learn Decision Trees:
  https://scikit-learn.org/stable/modules/tree.html
- scikit-learn Ensemble Methods:
  https://scikit-learn.org/stable/modules/ensemble.html
