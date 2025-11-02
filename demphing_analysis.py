import numpy as np
import matplotlib.pyplot as plt

def get_v_p(E: float, nu: float, rho: float):
    return np.sqrt(E * (1 - nu) / (1 + nu) / (1 - 2 * nu) / rho)
def get_v_s(E: float, nu: float, rho: float):
    return np.sqrt(E / 2 / (1 + nu) / rho)
def get_v_r(E: float, nu: float, rho: float):
    return (0.862 + 1.14 * nu) / (1 + nu) * get_v_s(E, nu, rho)

def demphing_analysis(tensor: np.ndarray, time_line: np.ndarray, size_block: int, T_exp: float, E: float, nu: float, rho: float, L: float, x_0: float=1, x_1: float=3, y_0: float=None, y_1: float=None, epsilon_order: float=1e-6, receiver_visibility: list[bool]=None, plot_title: str=None):
    print(L)
    v_p = get_v_p(E, nu, rho)
    print(f"Аналитическая скорость продольной волны: {v_p:.2f}")
    v_s = get_v_s(E, nu, rho)
    print(f"Аналитическая скорость поперечной волны: {v_s:.2f}")
    v_r = get_v_r(E, nu, rho)
    print(f"Аналитическая скорость Рэлеевской волны: {v_r:.2f}")

    print(f"Временной шаг: {time_line[1] - time_line[0]:.2e}")

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

    t_intersection_analytical = 2 * L / (v_p + v_s)
    print(f"Аналитическое время пересечения волн: {t_intersection_analytical:.2f} с")

    ind_intersection_analytical = np.argmin(np.abs(time_line - t_intersection_analytical))
    print(f"Аналитический индекс пересечения волн: {ind_intersection_analytical}")

    x_intersection_analytical = t_intersection_analytical * v_s
    print(f"Аналитическое расстояние пересечения волн: {x_intersection_analytical:.2f} м")

    reciever_intersection_analytical = np.argmin(np.abs(
        np.array([size_block * i for i in range(tensor.shape[0])]) - x_intersection_analytical)
    )
    print(f"Аналитический номер приемника, ближайшего к пересечению: {reciever_intersection_analytical}")

    t_intersection_exp = 2 * L / (v_p_exp + v_s)
    print(f"Экспериментальное время пересечения волн: {t_intersection_exp:.2f} с")

    ind_intersection_exp = np.argmin(np.abs(time_line - t_intersection_exp))
    print(f"Экспериментальный индекс пересечения волн: {ind_intersection_exp}")

    x_intersection_exp = t_intersection_exp * v_s
    print(f"Экспериментальное расстояние пересечения волн: {x_intersection_exp:.2f} м")

    reciever_intersection_exp = np.argmin(np.abs(
        np.array([size_block * i for i in range(tensor.shape[0])]) - x_intersection_exp)
    )
    print(f"Экспериментальный номер приемника, ближайшего к пересечению: {reciever_intersection_exp}")

    reciever_intersection = reciever_intersection_exp
    ind_intersection = ind_intersection_exp
    x_intersection = x_intersection_exp
    t_intersection = t_intersection_exp

    print('-'*20)

    print(f"Время прихода падающей волны через ресивер пересечения: {reciever_intersection * size_block / v_p_exp:.2f} с")

    ind_intersection_wave = np.argmin(np.abs(time_line - reciever_intersection * size_block / v_p_exp))
    print(f"Индекс прихода падающей волны через ресивер пересечения: {ind_intersection_wave}")

    reflection_wave_amplitude = np.abs(tensor[reciever_intersection:-1, ind_wave:ind_intersection]).max()
    print(f"Максимальная амплитуда отраженной волны: {reflection_wave_amplitude}")

    falling_wave_amplitude = np.abs(tensor[reciever_intersection:-1, ind_intersection_wave:ind_wave]).max()
    print(f"Максимальная амплитуда падающей волны: {falling_wave_amplitude}")

    print(f"Отношение амплитуд падающей и отраженной волны: {100 * falling_wave_amplitude / reflection_wave_amplitude:.2f}%")

    print('-'*20)
    recievers_number = tensor.shape[0]
    for reciever in range(reciever_intersection, recievers_number):
        try:
            if not receiver_visibility[reciever]:
                continue
            time_fall_wave = reciever * size_block / v_p_exp
            print(f"{time_fall_wave:.2f}:{t_intersection:.2f} с")
            ind_fall_wave = np.argmin(np.abs(time_line - time_fall_wave))
            amplitude_max = np.abs(
                tensor[reciever, ind_fall_wave:ind_intersection]
            ).max()
            
            time_refl_wave = (2 * L - reciever * size_block) / v_p_exp
            print(f"{time_refl_wave:.2f}:{t_intersection:.2f} с")
            ind_refl_wave = np.argmin(np.abs(time_line - time_refl_wave))
            amplitude_reflection = np.abs(
                tensor[reciever, ind_refl_wave:ind_intersection]
            ).max()
            
            print(f"{reciever}: Отношение амплитуд падающей ({amplitude_max:.3e}) и отраженной ({amplitude_reflection:.3e}) волны: {100 * amplitude_reflection / amplitude_max:.2f}%")
        except:
            if reciever == reciever_intersection:
                print(f"{reciever}: Продольная волна пересекласлась с поперечной")
            else:
                print(f"{reciever}: проблема с расчетом")
        
        print('-'*4)

    print('-'*20)

    a = np.abs(tensor[reciever_intersection:-1, np.argmin(np.abs(time_line - reciever_intersection * size_block / v_p_exp)):ind_wave]).max()
    b = np.abs(tensor[reciever_intersection:-1, ind_wave:ind_intersection]).max()
    print(f"Отношение амплитуд падающей ({a:.3e}) и отраженной ({b:.3e}) волны: {100 * b / a:.2f}%")

    print('-'*20)


    if receiver_visibility is None:
        receiver_visibility = [True for _ in range(recievers_number)]
    plt.figure(figsize=(18, 10))
    for reciever in range(reciever_intersection, recievers_number):
        if not receiver_visibility[reciever]:
            continue
        plt.plot(time_line, tensor[reciever, :], label=f"{reciever * size_block}")
        # plt.semilogy(times, np.abs(tensor[reciever, :]), label=f"{reciever * size_block}")
        print(f"{reciever * size_block}: t_p={reciever * size_block / v_p_exp} t_s={(2*L - reciever * size_block) / v_p_exp:.2f} t_s_exp={reciever * size_block / v_s_exp:.2f}")
        plt.axvline(x=reciever * size_block / v_p_exp, color='green', linestyle='--')
        plt.axvline(x=(2*L - reciever * size_block) / v_p_exp, color='blue', linestyle='--')

        plt.axvline(x=reciever * size_block / v_s_exp, color='yellow', linestyle='--')


    # plt.xlim(0, t_intersection)
    plt.axvline(x=t_intersection, color='red', linestyle='--')
    plt.axvline(x=time_line[ind_wave], color='black', linestyle='--')
    plt.xlim(x_0, x_1)
    if y_0 is not None and y_1 is not None:
        plt.ylim(y_0, y_1)
    plt.legend()
    if plot_title is not None:
        plt.title(plot_title)
    plt.show()