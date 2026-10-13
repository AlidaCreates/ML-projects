# Numeric Feature Engineering — California Housing

Продолжение Day 11 (piscine1b_w3). Проверяем, какие числовые
преобразования реально улучшают качество модели, а какие — нет,
и фиксируем каждое решение цифрой.

## Environment

Использует тот же venv, что и Day 11 — новых пакетов не требуется.

```bash
# активировать существующий venv
source piscine1b_w3/bin/activate      # Linux/macOS
piscine1b_w3\Scripts\activate         # Windows PowerShell

python numeric_fe.py
```

Зависимости: `pandas`, `numpy`, `scikit-learn` (>=1.0).

## Файлы

- `numeric_fe.py` — единый скрипт, Exercise 0–6, выполняется от начала
  до конца без ручных правок.
- `run_output.txt` — реальный вывод последнего запуска (для сверки).

## Что делает скрипт

| Exercise | Что проверяем | Результат (test R², LinearRegression) |
|---|---|---|
| 0 | Baseline (raw features) | **0.6139** |
| 1 | log1p на total_rooms/total_bedrooms/population/households | 0.6211 (**+0.0072**) |
| 2 | + ratio-фичи (rooms_per_household, bedrooms_per_room, population_per_household) | 0.5942 (**-0.0197** от baseline) |
| 3 | + PolynomialFeatures(degree=2) на 11 фичах → 77 фичей | LinReg: -2.0998 (взрыв), Ridge(alpha=1): 0.3559 |
| 4 | + KBinsDiscretizer(5 бинов) на housing_median_age | 0.5948 (**+0.0006** к ratio-набору) |
| 5 | То же самое, но RandomForestRegressor | raw 0.8101 → ratios 0.8021 (дерево почти не выигрывает от engineering) |
| 6 | Итоговая `engineer_features()` = log + ratios | LinReg 0.5942 (-0.0197), RF 0.8021 (-0.0080) |

## Решения (experiment log)

| step | test R² | kept |
|---|---|---|
| baseline | 0.6139 | — |
| + log | 0.6211 | **да** |
| + ratios | 0.5942 | нет |
| + poly(2) | -2.0998 (LinReg) / 0.3559 (Ridge) | нет |
| + age bin | 0.5948 | нет |

**Вывод:** в этот датасет единственное улучшение, которое реально
удерживается — log1p на скошенных count-фичах (+0.007). Ratio-фичи,
полином второй степени и биннинг возраста дома либо ухудшают линейную
модель, либо дают выигрыш меньше шума. `engineer_features()` в
Exercise 6 намеренно включает только log + ratios, как того требует
условие задания, но по цифрам из experiment log ratio-фичи стоило бы
исключить — это видно из таблицы и явно прокомментировано в выводе
скрипта.

Random Forest почти не реагирует ни на одно из преобразований
(дерево scale-invariant и само ловит нелинейности/взаимодействия) —
разница между raw и engineered на RF на порядок меньше, чем на
линейной модели.

## Проверка вывода

Все числа, напечатанные скриптом, совпадают с "Expected output" из
условия задания (Ex.0–Ex.5) — можно сверить построчно с
`run_output.txt`. Единственное расхождение — в самом условии
некоторые *текстовые* комментарии («delta vs log: +0.022» в Ex.2)
противоречат таблицам с числами в том же условии; скрипт всегда
считает и печатает фактические числа, а не переписывает текст
задания.
