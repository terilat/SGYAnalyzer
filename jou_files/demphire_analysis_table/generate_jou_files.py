#!/usr/bin/env python3
"""
Скрипт для генерации .jou файлов с разными значениями L, DW, R
"""

import os
import math
from pathlib import Path


def substitute_expressions(text, L, DW, R):
    """
    Заменяет выражения в фигурных скобках на вычисленные значения
    """
    # Заменяем основные выражения
    text = text.replace('{2*L}', str(2 * L))
    text = text.replace('{L - DW}', str(L - DW))
    text = text.replace('{L // 2}', str(L // 2))
    text = text.replace('{L}', str(L))
    text = text.replace('{DW}', str(DW))
    text = text.replace('{R}', str(R))
    
    return text


def calculate_R_values(L, DW, R, c_p, num_rows):
    """
    Вычисляет значения R_(ind_row) для таблиц
    
    R_(ind_row) = b_0 * ((L - DW + 10 * ind_row) / DW)**3
    b_0 = -4 * c_p * ln(R) / 2 / DW
    """
    if R < 1e-10:
        b_0 = 0.0
    else:
        b_0 = -4 * c_p * math.log(R) / 2 / DW
    R_values = []
    
    for ind_row in range(num_rows):
        x = 10 * ind_row
        R_val = b_0 * ((x / DW) ** 3)
        R_values.append(R_val)
    
    return R_values


def generate_template_2(L, DW, R, c_p):
    """
    Генерирует вторую часть файла с таблицами материалов
    
    Для каждой таблицы создается DW // 10 + 1 строк.
    Формула для первой колонки: L - DW + ind_row * 10, где ind_row от 0 до DW//10
    Формула для R: b_0 * ((L - DW + 10 * ind_row) / DW)**3
    где b_0 = -4 * c_p * ln(R) / 2 / DW
    """
    num_rows = DW // 10 + 1
    R_values = calculate_R_values(L, DW, R, c_p, num_rows)
    
    lines = []
    
    # Таблица 1 (левая сторона - значения с минусом)
    # Таблица уже создана в template_1, просто задаем dependency
    lines.append("modify table 1 dependency x")
    
    # Создание строк (вставляем строки сверху, поэтому делаем num_rows раз)
    for _ in range(num_rows):
        lines.append("modify table 1 insert row 1")
    
    # Заполнение первой колонки (с минусом)
    # ind_row в range(0, num_rows) соответствует строкам 1, 2, ..., num_rows
    for ind_row in range(num_rows):
        x_value = -(L - DW + 10 * ind_row)
        lines.append(f"modify table 1 cell {ind_row + 1} 1 value {x_value}")
    
    # Заполнение второй колонки
    for ind_row in range(num_rows):
        lines.append(f"modify table 1 cell {ind_row + 1} 2 value {R_values[ind_row]}")
    
    # Таблица 2 (правая сторона - значения с плюсом)
    lines.append("")
    lines.append("modify table 2 dependency x")
    
    # Создание строк
    for _ in range(num_rows):
        lines.append("modify table 2 insert row 1")
    
    # Заполнение первой колонки (с плюсом)
    for ind_row in range(num_rows):
        x_value = L - DW + 10 * ind_row
        lines.append(f"modify table 2 cell {ind_row + 1} 1 value {x_value}")
    
    # Заполнение второй колонки
    for ind_row in range(num_rows):
        lines.append(f"modify table 2 cell {ind_row + 1} 2 value {R_values[ind_row]}")
    
    # Таблица 3 (нижняя сторона - значения с минусом)
    lines.append("")
    lines.append("modify table 3 dependency y")
    
    # Создание строк
    for _ in range(num_rows):
        lines.append("modify table 3 insert row 1")
    
    # Заполнение первой колонки (с минусом)
    for ind_row in range(num_rows):
        y_value = -(L - DW + 10 * ind_row)
        lines.append(f"modify table 3 cell {ind_row + 1} 1 value {y_value}")
    
    # Заполнение второй колонки
    for ind_row in range(num_rows):
        lines.append(f"modify table 3 cell {ind_row + 1} 2 value {R_values[ind_row]}")
    
    return "\n".join(lines)


def generate_jou_file(L, DW, R, c_p, output_path, base_path):
    """
    Генерирует полный .jou файл из трех шаблонов
    """
    # Читаем шаблоны
    template_1_path = base_path / "template_1.jou"
    template_3_path = base_path / "template_3.jou"
    
    with open(template_1_path, 'r', encoding='utf-8') as f:
        template_1 = f.read()
    
    with open(template_3_path, 'r', encoding='utf-8') as f:
        template_3 = f.read()
    
    # Генерируем части
    part_1 = substitute_expressions(template_1, L, DW, R)
    part_2 = generate_template_2(L, DW, R, c_p)
    
    # Модифицируем последнюю строку template_3 с путем
    template_3_lines = template_3.split('\n')
    if template_3_lines:
        # Заменяем последнюю строку
        last_line = template_3_lines[-1]
        if last_line.startswith("calculation start path"):
            # Формируем путь: 'B:\ELfimov\Dissertation\exps\Lamb_demphire\lamb_demphire_L_{L}_DW_{DW}_R_{R}_20_7\output.pvd'
            output_dir = f"lamb_demphire_L_{L}_DW_{DW}_R_{R}_20_7"
            path_str = f"calculation start path 'B:\\ELfimov\\Dissertation\\exps\\Lamb_demphire\\{output_dir}\\output.pvd'"
            template_3_lines[-1] = path_str
        else:
            # Если последняя строка не та, добавляем новую
            output_dir = f"lamb_demphire_L_{L}_DW_{DW}_R_{R}_20_7"
            path_str = f"calculation start path 'B:\\ELfimov\\Dissertation\\exps\\Lamb_demphire\\{output_dir}\\output.pvd'"
            template_3_lines.append(path_str)
    
    part_3 = "\n".join(template_3_lines)
    
    # Объединяем все части
    full_content = part_1 + "\n\n" + part_2 + "\n\n" + part_3
    
    # Сохраняем файл
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_content)
    
    print(f"Generated: {output_path}")


def main():
    """
    Основная функция для генерации файлов
    """
    # Параметры для генерации
    # Здесь задайте нужные значения L, DW, R и c_p
    
    # Пример: списки значений для перебора
    L_values = [500, 640, 760]  # Пример значений
    DW_values = [100, 160, 240]  # Пример значений
    R_values = [0.0, 0.1, 0.01, 0.001, 0.0001]  # Пример значений
    c_p = 376.43  # Скорость волны P (можно задать как константу или список)
    
    # Путь к директории с шаблонами
    script_dir = Path(__file__).parent
    base_path = script_dir
    
    # Создаем директорию для выходных файлов
    output_dir = script_dir / "generated"
    output_dir.mkdir(exist_ok=True)
    
    # Генерируем файлы для всех комбинаций
    for L in L_values:
        for DW in DW_values:
            for R in R_values:
                # Формируем имя файла
                filename = f"lamb_demphire_L_{L}_DW_{DW}_R_{R}_20_7.jou"
                output_path = output_dir / filename
                
                # Генерируем файл
                generate_jou_file(L, DW, R, c_p, output_path, base_path)
    
    print(f"\nGeneration complete! Files saved to: {output_dir}")


if __name__ == "__main__":
    main()

