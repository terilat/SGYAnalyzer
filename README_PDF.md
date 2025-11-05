# PDF Generator для анализа затухания волн

## Описание

Модуль `demphing_analysis_pdf.py` - автоматическая генерация PDF отчетов по анализу затухания волн **БЕЗ вывода в консоль**. Только чистая генерация PDF.

## Структура выходных файлов

```
pdfs/
├── figures/          # PNG изображения графиков
│   ├── {name}_tensor_imshow.png
│   └── {name}_waves_analysis.png
├── texs/            # LaTeX исходники
│   └── {name}.tex
└── pdf/             # Скомпилированные PDF
    └── {name}.pdf
```

## Использование

```python
from demphing_analysis_pdf import demphing_analysis_pdf

# Ваши данные
tensor = ...  # np.ndarray
time_line = ...  # np.ndarray

# Генерация PDF
amplitudes, pdf_path = demphing_analysis_pdf(
    tensor=tensor,
    time_line=time_line,
    size_block=20,
    T_exp=5.0,
    E=1e9,
    nu=0.25,
    rho=2000,
    L=640,
    DW=100,
    R=0.01,
    x_0=1,
    x_1=3,
    y_0=-2e-5,
    y_1=2e-5,
    plot_title="Анализ затухания волн",
    output_name="my_analysis"  # Опционально (по умолчанию - timestamp)
)

print(f"PDF создан: {pdf_path}")
```

## Параметры

### Обязательные:
- `tensor` - массив данных волн
- `time_line` - временная шкала
- `size_block` - расстояние между приемниками (м)
- `T_exp` - полное время эксперимента (с)
- `E` - модуль Юнга (Па)
- `nu` - коэффициент Пуассона
- `rho` - плотность (кг/м³)
- `L` - длина образца (м)
- `DW` - ширина образца (м)
- `R` - коэффициент затухания

### Опциональные:
- `x_0, x_1` - пределы оси X для графиков (по умолчанию 1, 3)
- `y_0, y_1` - пределы оси Y (опционально)
- `epsilon_order` - порядок величины ошибки (по умолчанию 1e-6)
- `receiver_visibility` - список видимости приемников
- `plot_title` - заголовок отчета
- `output_name` - имя выходных файлов (по умолчанию - timestamp)

## Возвращаемые значения

- `amplitudes` (dict) - результаты анализа амплитуд
- `pdf_path` (str) - путь к созданному PDF файлу

## Содержание PDF отчета

1. **Параметры моделирования** - L, DW, R
2. **Расчётные скорости волн** - vₚ, vₛ, vᵣ
3. **Анализ прихода продольной волны**
4. **Анализ пересечения волн**
5. **График распространения волн** (tensor imshow)
6. **График анализа волн** (waves analysis)
7. **Таблица результатов амплитуд** - для всех приемников

## Требования

- Python 3.7+
- numpy
- matplotlib
- pdflatex (texlive)

### Установка pdflatex

**Ubuntu/Debian:**
```bash
sudo apt-get install texlive-latex-base texlive-lang-cyrillic texlive-latex-extra
```

## Пример

См. `example_pdf_generation.py`

## Особенности

- ✅ Нет вывода в консоль (только генерация PDF)
- ✅ Автоматическое создание структуры папок
- ✅ Шаблон LaTeX с подстановкой значений через f-строки
- ✅ Автоматическая очистка вспомогательных файлов (.aux, .log)
- ✅ Таблица с результатами анализа амплитуд
- ✅ Два графика высокого разрешения (300 dpi)

