"""
Пример использования demphing_analysis_pdf для генерации PDF отчета
"""

import numpy as np
from demphing_analysis_pdf import demphing_analysis_pdf
import os
from SGYReader import get_seysmic_tensor

sgy_files = os.listdir('sgy_files/demphire_coord_table')
len(sgy_files)

params = []
for file in sgy_files:
    param = file.split('_')
    params.append({
        'L': int(param[3]),
        'DW': int(param[5]),
        'R': float(param[7])
    })

params.sort(key=lambda x: f"{x['L']}_{x['DW']}_{x['R']:.2e}")
for i in range(len(params)):
    print(f'{i}: {params[i]}')

tensors_list = []
time_lines_list = []
time_evaluations_list = []
for param in params:
    tensor, time_line, time_evaluation = get_seysmic_tensor(os.path.join('sgy_files', 'demphire_coord_table'), exp=f'lamb_demphire_L_{param["L"]}_DW_{param["DW"]}_R_{param["R"]}_20_7')
    tensors_list.append(tensor)
    time_lines_list.append(time_line)
    time_evaluations_list.append(time_evaluation)
    if time_evaluation is None:
        print(f"Time evaluation is None for {param}")

# Генерация PDF отчета
for ind_plot in range(len(params)):
    L, DW, R = params[ind_plot]['L'], params[ind_plot]['DW'], params[ind_plot]['R']
    print(f"Start analysis for L={L}, DW={DW}, R={R}")

    receiver_visibility = np.array([False for _ in range(tensors_list[ind_plot]['Uy'].shape[0])])
    receiver_visibility[-3] = True

    demphing_analysis_pdf(
        tensor=tensors_list[ind_plot]['Uy'],
        time_line=time_lines_list[ind_plot],
        size_block=20,
        T_exp=3,
        E=2e+8,
        nu=0.3,
        rho=1900,
        L=L,
        DW=DW,
        R=R,
        x_0=0.8,
        x_1=3,
        y_0=-1e-3,
        y_1=1e-3,
        epsilon_order=np.abs(tensors_list[ind_plot]['Uy']).max() / 1e2,
        receiver_visibility=receiver_visibility,
        plot_title=f"Анализ затухания волн: L={L}, DW={DW}, R={R}",
        output_name=f"wave_analysis_example_{L}_{DW}_{R}"  # Имя выходных файлов
    )

    print(f"End analysis for L={L}, DW={DW}, R={R}")
