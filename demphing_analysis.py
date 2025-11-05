import numpy as np
import matplotlib.pyplot as plt
from SGYReader import local_extremums, find_border

def get_v_p(E: float, nu: float, rho: float):
    return np.sqrt(E * (1 - nu) / (1 + nu) / (1 - 2 * nu) / rho)
def get_v_s(E: float, nu: float, rho: float):
    return np.sqrt(E / 2 / (1 + nu) / rho)
def get_v_r(E: float, nu: float, rho: float):
    return (0.862 + 1.14 * nu) / (1 + nu) * get_v_s(E, nu, rho)

def demphing_analysis(tensor: np.ndarray, time_line: np.ndarray, size_block: int, T_exp: float, E: float, nu: float, rho: float, L: float, DW: float, R: float, x_0: float=1, x_1: float=3, y_0: float=None, y_1: float=None, epsilon_order: float=1e-6, receiver_visibility: list[bool]=None, plot_title: str=None, verbose: bool=True):
    v_p = get_v_p(E, nu, rho)
    print(f"Аналитическая скорость продольной волны: {v_p:.2f}")
    v_s = get_v_s(E, nu, rho)
    print(f"Аналитическая скорость поперечной волны: {v_s:.2f}")
    v_r = get_v_r(E, nu, rho)
    print(f"Аналитическая скорость Рэлеевской волны: {v_r:.2f}")

    print(f"Временной шаг: {time_line[1] - time_line[0]:.2e}")

    v_s_exp = 250
    v_p_ref = 192.25
    print(f"Экспериментальная скорость поперечной волны: {v_s_exp:.2f} м/с")
    print(f"Экспериментальная скорость отраженной продольной волны: {v_p_ref:.2f} м/с")

    eps = np.abs(tensor[:, -1]).max() * epsilon_order
    print(f"Величина ошибки: {eps:.2e}")

    print('-'*20)

    print(f"Аналитическое время прихода продольной волны: {L / v_p:.2f} с")

    ind_wave = np.argmax(np.abs(tensor[-1, :]) > eps) 
    # ind_wave = np.argmin(np.abs(time_line - L / v_p))
    print(f"Экспериментальный индекс прихода продольной волны: {ind_wave}")
    print(f"Экспериментальное время прихода продольной волны: {time_line[ind_wave]:.2f} с")
    
    v_p_exp = L / time_line[ind_wave]
    print(f"Скорость пришедшей продольной волны: {v_p_exp:.2f} м/с")

    length_inds_wave_p = np.argmin(np.abs(tensor[-1, ind_wave + 50:] - tensor[-1, ind_wave]))
    print(f"Длины продольной волны в индексах: {length_inds_wave_p}")
    length_meters_wave_p = length_inds_wave_p / tensor.shape[-1] * T_exp * v_p_exp
    print(f"Длина продольной волны в метрах: {length_meters_wave_p:.2f} м")

    print('-'*20)

    t_intersection = L / v_p_exp + (L - L / v_p_exp * v_s_exp) / (v_s_exp + v_p_ref) #2 * L / (v_p_exp + v_s)
    print(f"Экспериментальное время пересечения волн: {t_intersection:.2f} с")

    ind_intersection = np.argmin(np.abs(time_line - t_intersection))
    print(f"Экспериментальный индекс пересечения волн: {ind_intersection}")

    x_intersection = t_intersection * v_s_exp
    print(f"Экспериментальное расстояние пересечения волн: {x_intersection:.2f} м")

    reciever_intersection = np.argmin(np.abs(
        np.array([size_block * i for i in range(tensor.shape[0])]) - x_intersection)
    )
    print(f"Экспериментальный номер приемника, ближайшего к пересечению: {reciever_intersection}")

    print('-'*20)

    print(f"Время прихода падающей волны через ресивер пересечения: {reciever_intersection * size_block / v_p_exp:.2f} с")

    ind_intersection_wave = np.argmin(np.abs(time_line - reciever_intersection * size_block / v_p_exp))
    print(f"Индекс прихода падающей волны через ресивер пересечения: {ind_intersection_wave}")

    print('-'*20)

    ### IMSHOW PLOT
    if verbose:
        imshow_tensor(tensor, size_block, T_exp, v_p_exp, L, DW, R, t_intersection, reciever_intersection)

    ### ANALYSIS OF REFLECTION AND FALLING WAVES
    recievers_number = tensor.shape[0]    
    if receiver_visibility is None:
        receiver_visibility = [True for _ in range(recievers_number)]
    
    if verbose:
        plots_visualize(tensor, time_line, size_block, T_exp, v_p_exp, v_s_exp, v_p_ref, L, DW, R, ind_intersection, t_intersection, reciever_intersection, recievers_number, receiver_visibility, ind_wave, x_0, x_1, y_0, y_1, plot_title)

    amplitudes = {}

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

        print(f"{reciever * size_block}: t_p_fall={t_p_fall:.3f} t_p_refl={t_p_refl:.3f} t_s_fall={t_s_fall:.3f} | t_ints={t_intersection:.3f}")
        print(f"amplitude_p_fall={amplitude_p_fall:.3e} amplitude_p_refl_0={amplitude_p_refl_0:.3e} | amplitude_p_refl_1={amplitude_p_refl_1:.3e}")
        print(f"ratio_0={100 * amplitude_p_refl_0 / amplitude_p_fall:.2f}% ratio_1={100 * amplitude_p_refl_1 / amplitude_p_fall:.2f}%")
        print('-'*4)

    return amplitudes


def imshow_tensor(tensor: np.ndarray, size_block: int, T_exp: float, v_p_exp: float, L: float, DW: float, R: float, t_intersection: float, reciever_intersection: int):
    extent = [0, T_exp, L, 0]

    plt.figure(figsize=(16, 10))
    plt.imshow(
        tensor, 
        aspect=0.004, 
        vmin=-1e-5, 
        vmax=1e-5,
        extent=extent
    )
    plt.title(f'L={L}, DW={DW}, R={R}, t_intersection={t_intersection:.2f}')

    # plt.axvline(x=left_time, color='green', linestyle='--')
    # plt.axvline(x=right_time, color='green', linestyle='--')
    plt.axvline(x=t_intersection, color='green', linestyle='--')
    for receiver in range(reciever_intersection - 1, tensor.shape[0] - 1):
        plt.axhline(y=size_block * receiver, color='green', linestyle='--')
    N = tensor.shape[-1]

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
    plt.show()


def plots_visualize(
    tensor: np.ndarray, 
    time_line: np.ndarray, 
    size_block: int, 
    T_exp: float, 
    v_p_exp: float, 
    v_s_exp: float, 
    v_p_ref: float, 

    L: float, 
    DW: float, 
    R: float, 
    ind_intersection: int,
    t_intersection: float, 
    
    reciever_intersection: int, 
    recievers_number: int, 
    receiver_visibility: list[bool], 

    ind_wave: int, 

    x_0: float=1, 
    x_1: float=3, 
    y_0: float=None, 
    y_1: float=None, 
    plot_title: str=None
):

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

        plt.plot([amplitude_p_fall_t, amplitude_p_fall_t], [0, tensor[reciever, amplitude_p_fall_ind]], color='purple', linestyle='-', linewidth=2)
        plt.plot([amplitude_p_refl_0_t, amplitude_p_refl_0_t], [0, tensor[reciever, amplitude_p_refl_0_ind]], color='purple', linestyle='-', linewidth=2)
        plt.plot([amplitude_p_refl_1_t, amplitude_p_refl_1_t], [0, tensor[reciever, amplitude_p_refl_1_ind]], color='purple', linestyle='-', linewidth=2)


    # plt.xlim(0, t_intersection)
    plt.axvline(x=t_intersection, color='red', linestyle='--')
    plt.axvline(x=time_line[ind_wave], color='black', linestyle='--')
    plt.xlim(x_0, x_1)
    plt.plot([x_0, x_1], [0, 0], color='black', linestyle='--', linewidth=1)
    if y_0 is not None and y_1 is not None:
        plt.ylim(y_0, y_1)
    plt.legend()
    if plot_title is not None:
        plt.title(plot_title)
    plt.show()