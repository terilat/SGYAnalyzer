import numpy as np
import matplotlib.pyplot as plt
from SGYReader import local_extremums, find_border
import subprocess
from pathlib import Path
from datetime import datetime

def get_v_p(E: float, nu: float, rho: float):
    return np.sqrt(E * (1 - nu) / (1 + nu) / (1 - 2 * nu) / rho)

def get_v_s(E: float, nu: float, rho: float):
    return np.sqrt(E / 2 / (1 + nu) / rho)

def get_v_r(E: float, nu: float, rho: float):
    return (0.862 + 1.14 * nu) / (1 + nu) * get_v_s(E, nu, rho)


def demphing_analysis_pdf(
    tensor: np.ndarray, 
    time_line: np.ndarray, 
    size_block: int, 
    T_exp: float, 
    E: float, 
    nu: float, 
    rho: float, 
    L: float, 
    DW: float, 
    R: float, 
    x_0: float=1, 
    x_1: float=3, 
    y_0: float=None, 
    y_1: float=None, 
    epsilon_order: float=1e-6, 
    receiver_visibility: list[bool]=None, 
    plot_title: str=None,
    output_name: str=None
):
    """
    Generate PDF report with wave analysis.
    
    Args:
        tensor: Wave data tensor
        time_line: Time line array
        size_block: Distance between receivers in meters
        T_exp: Total experiment time
        E: Young's modulus (Pa)
        nu: Poisson's ratio
        rho: Density (kg/m³)
        L: Sample length (m)
        DW: Sample width (m)
        R: Damping coefficient
        x_0, x_1: X-axis limits for plots
        y_0, y_1: Y-axis limits for plots
        epsilon_order: Error magnitude order
        receiver_visibility: List of receiver visibility flags
        plot_title: Title for the report
        output_name: Name for output files (default: timestamp)
    
    Returns:
        dict: Amplitude analysis results
        str: Path to generated PDF
    """
    
    # Setup output directories
    base_dir = Path("./pdfs")
    figures_dir = base_dir / "figures"
    texs_dir = base_dir / "texs"
    pdf_dir = base_dir / "pdf"
    
    for dir_path in [figures_dir, texs_dir, pdf_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    if output_name is None:
        output_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate velocities
    v_p = get_v_p(E, nu, rho)
    v_s = get_v_s(E, nu, rho)
    v_r = get_v_r(E, nu, rho)
    
    time_step = time_line[1] - time_line[0]
    
    v_s_exp = 250
    v_p_ref = 192.25
    
    eps = np.abs(tensor[:, -1]).max() * epsilon_order
    
    # Wave arrival calculations
    t_analytical = L / v_p
    ind_wave = np.argmax(np.abs(tensor[-1, :]) > eps)
    t_wave = time_line[ind_wave]
    v_p_exp = L / time_line[ind_wave]
    
    length_inds_wave_p = np.argmin(np.abs(tensor[-1, ind_wave + 50:] - tensor[-1, ind_wave]))
    length_meters_wave_p = length_inds_wave_p / tensor.shape[-1] * T_exp * v_p_exp
    
    # Intersection calculations
    t_intersection = L / v_p_exp + (L - L / v_p_exp * v_s_exp) / (v_s_exp + v_p_ref)
    ind_intersection = np.argmin(np.abs(time_line - t_intersection))
    x_intersection = t_intersection * v_s_exp
    
    reciever_intersection = np.argmin(np.abs(
        np.array([size_block * i for i in range(tensor.shape[0])]) - x_intersection)
    )
    
    ind_intersection_wave = np.argmin(np.abs(time_line - reciever_intersection * size_block / v_p_exp))
    
    # Generate figures
    img1_path = figures_dir / f"{output_name}_tensor_imshow.png"
    img2_path = figures_dir / f"{output_name}_waves_analysis.png"
    
    generate_tensor_plot(tensor, size_block, T_exp, v_p_exp, L, DW, R, 
                        t_intersection, reciever_intersection, img1_path)
    
    # Analysis of reflection and falling waves
    recievers_number = tensor.shape[0]
    if receiver_visibility is None:
        receiver_visibility = [True for _ in range(recievers_number)]
    
    generate_waves_plot(tensor, time_line, size_block, T_exp, v_p_exp, v_s_exp, v_p_ref,
                       L, DW, R, ind_intersection, t_intersection, reciever_intersection,
                       recievers_number, receiver_visibility, ind_wave, x_0, x_1, y_0, y_1,
                       plot_title, img2_path)
    
    # Calculate amplitudes
    amplitudes = {}
    amplitude_data = []
    
    for reciever in range(reciever_intersection, recievers_number):
        if not receiver_visibility[reciever]:
            continue
        
        t_p_fall = reciever * size_block / v_p_exp
        t_p_refl = L / v_p_exp + (L - reciever * size_block) / v_p_ref
        t_s_fall = reciever * size_block / v_s_exp
        
        ind_p_fall = np.argmin(np.abs(time_line - t_p_fall))
        ind_p_refl = np.argmin(np.abs(time_line - t_p_refl))
        ind_s_fall = np.argmin(np.abs(time_line - t_s_fall))
        
        amplitude_p_fall = np.abs(tensor[reciever, ind_p_fall:ind_intersection]).max()
        amplitude_p_refl_0 = np.abs(tensor[reciever, ind_p_refl:ind_intersection]).max()
        amplitude_p_refl_1 = np.abs(tensor[reciever, ind_p_refl:ind_s_fall]).max()
        
        amplitudes[reciever] = {
            't_p_fall': t_p_fall,
            't_p_refl': t_p_refl,
            't_s_fall': t_s_fall,
            'amplitude_p_fall': amplitude_p_fall,
            'amplitude_p_refl_0': amplitude_p_refl_0,
            'amplitude_p_refl_1': amplitude_p_refl_1
        }
        
        ratio_0 = 100 * amplitude_p_refl_0 / amplitude_p_fall if amplitude_p_fall > 0 else 0
        ratio_1 = 100 * amplitude_p_refl_1 / amplitude_p_fall if amplitude_p_fall > 0 else 0
        
        amplitude_data.append({
            'position': reciever * size_block,
            't_p_fall': t_p_fall,
            't_p_refl': t_p_refl,
            't_s_fall': t_s_fall,
            't_intersection': t_intersection,
            'amplitude_p_fall': amplitude_p_fall,
            'amplitude_p_refl_0': amplitude_p_refl_0,
            'amplitude_p_refl_1': amplitude_p_refl_1,
            'ratio_0': ratio_0,
            'ratio_1': ratio_1
        })
    
    # Generate LaTeX document
    tex_content = generate_latex_template(
        L=L, DW=DW, R=R,
        v_p=v_p, v_s=v_s, v_r=v_r,
        time_step=time_step,
        v_s_exp=v_s_exp, v_p_ref=v_p_ref,
        eps=eps,
        t_analytical=t_analytical,
        ind_wave=ind_wave, t_wave=t_wave, v_p_exp=v_p_exp,
        length_inds_wave_p=length_inds_wave_p,
        length_meters_wave_p=length_meters_wave_p,
        t_intersection=t_intersection,
        ind_intersection=ind_intersection,
        x_intersection=x_intersection,
        reciever_intersection=reciever_intersection,
        ind_intersection_wave=ind_intersection_wave,
        amplitude_data=amplitude_data,
        img1_name=f"{output_name}_tensor_imshow.png",
        img2_name=f"{output_name}_waves_analysis.png",
        plot_title=plot_title
    )
    
    tex_path = texs_dir / f"{output_name}.tex"
    tex_path.write_text(tex_content, encoding='utf-8')
    
    # Compile to PDF
    pdf_path = compile_latex(tex_path, pdf_dir, output_name)
    
    return amplitudes, str(pdf_path)


def generate_tensor_plot(tensor, size_block, T_exp, v_p_exp, L, DW, R, 
                         t_intersection, reciever_intersection, save_path):
    """Generate and save tensor visualization plot."""
    extent = [0, T_exp, L, 0]
    
    plt.figure(figsize=(16, 10))
    plt.imshow(tensor, aspect=0.004, vmin=-1e-5, vmax=1e-5, extent=extent)
    plt.title(f'L={L}, DW={DW}, R={R}, t_intersection={t_intersection:.2f}')
    
    plt.axvline(x=t_intersection, color='green', linestyle='--')
    for receiver in range(reciever_intersection - 1, tensor.shape[0] - 1):
        plt.axhline(y=size_block * receiver, color='green', linestyle='--')
    
    f_x1 = 0
    f_x2 = L / v_p_exp
    f_y1 = 0
    f_y2 = 20 * (tensor.shape[0] - 1)
    plt.plot([f_x1, f_x2], [f_y1, f_y2], color='red', linestyle='-', linewidth=2)
    
    r_x1 = f_x2
    r_x2 = 3
    r_y1 = f_y2
    r_y2 = L - (3 - r_x1) * 192.25
    plt.plot([r_x1, r_x2], [r_y1, r_y2], color='red', linestyle='-', linewidth=2)
    
    plt.colorbar()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_waves_plot(tensor, time_line, size_block, T_exp, v_p_exp, v_s_exp, v_p_ref,
                       L, DW, R, ind_intersection, t_intersection, reciever_intersection,
                       recievers_number, receiver_visibility, ind_wave, x_0, x_1, y_0, y_1,
                       plot_title, save_path):
    """Generate and save waves analysis plot."""
    plt.figure(figsize=(18, 10))
    
    for reciever in range(reciever_intersection, recievers_number):
        if not receiver_visibility[reciever]:
            continue
        
        plt.plot(time_line, tensor[reciever, :], label=f"{reciever * size_block}")
        
        t_p_fall = reciever * size_block / v_p_exp
        t_p_refl = L / v_p_exp + (L - reciever * size_block) / v_p_ref
        t_s_fall = reciever * size_block / v_s_exp
        
        plt.axvline(x=t_p_fall, color='green', linestyle='--')
        plt.axvline(x=t_p_refl, color='blue', linestyle='--')
        plt.axvline(x=t_s_fall, color='yellow', linestyle='--')
        
        ind_p_fall = np.argmin(np.abs(time_line - t_p_fall))
        ind_p_refl = np.argmin(np.abs(time_line - t_p_refl))
        ind_s_fall = np.argmin(np.abs(time_line - t_s_fall))
        
        amplitude_p_fall_ind = np.abs(tensor[reciever, ind_p_fall:ind_intersection]).argmax() + ind_p_fall
        amplitude_p_refl_0_ind = np.abs(tensor[reciever, ind_p_refl:ind_intersection]).argmax() + ind_p_refl
        amplitude_p_refl_1_ind = np.abs(tensor[reciever, ind_p_refl:ind_s_fall]).argmax() + ind_p_refl
        
        amplitude_p_fall_t = amplitude_p_fall_ind / len(time_line) * T_exp
        amplitude_p_refl_0_t = amplitude_p_refl_0_ind / len(time_line) * T_exp
        amplitude_p_refl_1_t = amplitude_p_refl_1_ind / len(time_line) * T_exp
        
        plt.plot([amplitude_p_fall_t, amplitude_p_fall_t], [0, tensor[reciever, amplitude_p_fall_ind]], 
                color='purple', linestyle='-', linewidth=2)
        plt.plot([amplitude_p_refl_0_t, amplitude_p_refl_0_t], [0, tensor[reciever, amplitude_p_refl_0_ind]], 
                color='purple', linestyle='-', linewidth=2)
        plt.plot([amplitude_p_refl_1_t, amplitude_p_refl_1_t], [0, tensor[reciever, amplitude_p_refl_1_ind]], 
                color='purple', linestyle='-', linewidth=2)
    
    plt.axvline(x=t_intersection, color='red', linestyle='--')
    plt.axvline(x=time_line[ind_wave], color='black', linestyle='--')
    plt.xlim(x_0, x_1)
    plt.plot([x_0, x_1], [0, 0], color='black', linestyle='--', linewidth=1)
    
    if y_0 is not None and y_1 is not None:
        plt.ylim(y_0, y_1)
    
    plt.legend()
    if plot_title is not None:
        plt.title(plot_title)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_latex_template(L, DW, R, v_p, v_s, v_r, time_step, v_s_exp, v_p_ref, eps,
                            t_analytical, ind_wave, t_wave, v_p_exp, length_inds_wave_p,
                            length_meters_wave_p, t_intersection, ind_intersection,
                            x_intersection, reciever_intersection, ind_intersection_wave,
                            amplitude_data, img1_name, img2_name, plot_title=None):
    """Generate LaTeX document template with calculated values."""
    
    title = plot_title if plot_title else f"Wave Analysis: L={L}, DW={DW}, R={R}"
    
    # Build amplitude table rows
    amplitude_rows = ""
    for data in amplitude_data:
        amplitude_rows += f"""{data['position']:.1f} & {data['t_p_fall']:.3f} & {data['t_p_refl']:.3f} & {data['t_s_fall']:.3f} & {data['t_intersection']:.3f} & {data['amplitude_p_fall']:.3e} & {data['amplitude_p_refl_0']:.3e} & {data['amplitude_p_refl_1']:.3e} & {data['ratio_0']:.2f}\% & {data['ratio_1']:.2f}\% \\\\
"""
    
    latex_doc = r"""\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[russian]{babel}
\usepackage{graphicx}
\usepackage{float}
\usepackage{geometry}
\geometry{margin=2cm}
\usepackage{booktabs}
\usepackage{longtable}

\begin{document}

\section*{""" + title + r"""}

\subsection*{Параметры моделирования}

\begin{itemize}
    \item Длина образца: $L = """ + f"{L}" + r"""$ м
    \item Ширина образца: $DW = """ + f"{DW}" + r"""$ м
    \item Коэффициент затухания: $R = """ + f"{R}" + r"""$
    \item Модуль Юнга: $E$, Коэффициент Пуассона: $\nu$, Плотность: $\rho$
\end{itemize}

\subsection*{Расчётные скорости волн}

\begin{itemize}
    \item Аналитическая скорость продольной волны: $v_p = """ + f"{v_p:.2f}" + r"""$ м/с
    \item Аналитическая скорость поперечной волны: $v_s = """ + f"{v_s:.2f}" + r"""$ м/с
    \item Аналитическая скорость Рэлеевской волны: $v_r = """ + f"{v_r:.2f}" + r"""$ м/с
    \item Временной шаг: $\Delta t = """ + f"{time_step:.2e}" + r"""$ с
    \item Экспериментальная скорость поперечной волны: $v_s^{exp} = """ + f"{v_s_exp:.2f}" + r"""$ м/с
    \item Экспериментальная скорость отраженной продольной волны: $v_p^{ref} = """ + f"{v_p_ref:.2f}" + r"""$ м/с
    \item Величина ошибки: $\varepsilon = """ + f"{eps:.2e}" + r"""$
\end{itemize}

\subsection*{Анализ прихода продольной волны}

\begin{itemize}
    \item Аналитическое время прихода продольной волны: $t_{analytical} = """ + f"{t_analytical:.2f}" + r"""$ с
    \item Экспериментальный индекс прихода продольной волны: """ + f"{ind_wave}" + r"""
    \item Экспериментальное время прихода продольной волны: $t_{wave} = """ + f"{t_wave:.2f}" + r"""$ с
    \item Скорость пришедшей продольной волны: $v_p^{exp} = """ + f"{v_p_exp:.2f}" + r"""$ м/с
    \item Длина продольной волны в индексах: """ + f"{length_inds_wave_p}" + r"""
    \item Длина продольной волны в метрах: $\lambda_p = """ + f"{length_meters_wave_p:.2f}" + r"""$ м
\end{itemize}

\subsection*{Анализ пересечения волн}

\begin{itemize}
    \item Экспериментальное время пересечения волн: $t_{intersection} = """ + f"{t_intersection:.2f}" + r"""$ с
    \item Экспериментальный индекс пересечения волн: """ + f"{ind_intersection}" + r"""
    \item Экспериментальное расстояние пересечения волн: $x_{intersection} = """ + f"{x_intersection:.2f}" + r"""$ м
    \item Номер приемника, ближайшего к пересечению: """ + f"{reciever_intersection}" + r"""
    \item Индекс прихода падающей волны через ресивер пересечения: """ + f"{ind_intersection_wave}" + r"""
\end{itemize}

\subsection*{Визуализация тензора}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{./pdfs/figures/""" + img1_name + r"""}
    \caption{Визуализация распространения волн в среде}
    \label{fig:tensor}
\end{figure}

\subsection*{Анализ волн}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{./pdfs/figures/""" + img2_name + r"""}
    \caption{Анализ падающих и отраженных волн}
    \label{fig:waves}
\end{figure}

\subsection*{Результаты анализа амплитуд}

\begin{longtable}{|c|c|c|c|c|c|c|c|c|c|}
\hline
Позиция & $t_{p,fall}$ & $t_{p,refl}$ & $t_{s,fall}$ & $t_{int}$ & $A_{p,fall}$ & $A_{p,refl,0}$ & $A_{p,refl,1}$ & Ratio 0 & Ratio 1 \\
(м) & (с) & (с) & (с) & (с) & & & & (\%) & (\%) \\
\hline
\endfirsthead
\hline
Позиция & $t_{p,fall}$ & $t_{p,refl}$ & $t_{s,fall}$ & $t_{int}$ & $A_{p,fall}$ & $A_{p,refl,0}$ & $A_{p,refl,1}$ & Ratio 0 & Ratio 1 \\
\hline
\endhead
\hline
\endfoot
""" + amplitude_rows + r"""
\hline
\end{longtable}

\end{document}
"""
    
    return latex_doc


def compile_latex(tex_path: Path, pdf_dir: Path, output_name: str):
    """Compile LaTeX file to PDF using pdflatex."""
    
    # Run pdflatex twice for proper references
    for i in range(2):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(pdf_dir), str(tex_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    
    # Check if PDF was created
    pdf_path = pdf_dir / f"{output_name}.pdf"
    
    if pdf_path.exists():
        # Clean up auxiliary files
        for ext in ['.aux', '.log', '.out']:
            aux_file = pdf_dir / f"{output_name}{ext}"
            if aux_file.exists():
                aux_file.unlink()
        return pdf_path
    else:
        log_file = pdf_dir / f"{output_name}.log"
        raise RuntimeError(f"Failed to create PDF. Check {log_file} for details.")
