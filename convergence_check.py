import os
import numpy as np
import re
import matplotlib.pyplot as plt
from SGYReader import SGYReader
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConvergenceCheck:
    def __init__(self):
        pass
    
    

    # def plot(self, sizes_block: list[int], receivers: list[int], orders: list[int]):
    def plot(self, sizes_block, receivers, orders):
        fig, axs = plt.subplots(3, 2, figsize=(16, 12))

        for ind_variable, var in enumerate(['ux', 'uy', 'vx', 'vy', 'ax', 'ay']):
            for ind_reciever, size_block in enumerate(sizes_block): 
                for order in orders:
                # for order in [7, 8, 9]:
                    axs[ind_variable // 2][ind_variable % 2].plot(
                        self.x_ax[size_block][order], 
                        self.tensors_list[size_block][order][ind_variable, receivers[ind_reciever]], 
                        label=f'{size_block} {order} {size_block * receivers[ind_reciever]}'
                    )
                axs[ind_variable // 2][ind_variable % 2].set_title(f"Size block: {size_block}, variable: {var}")
                axs[ind_variable // 2][ind_variable % 2].legend()

    # def error(self, size_block: int, order_1: int, order_2: int, recievers: list[int]):
    def error(self, size_block, order_1, order_2, recievers):
        # errors = defaultdict(dict)
        errors = {}
        errors[size_block] = {}
        for var in ['u_x', 'u_y', 'v_x', 'v_y', 'a_x', 'a_y']:
            errors[size_block][var] = []

        for reciever in recievers:
            t_a = self.x_ax[size_block][order_1]
            t_b = self.x_ax[size_block][order_2]
            
            print(t_a.shape, t_a[-1])
            print(t_b.shape, t_b[-1])
            for ind, var in enumerate(['u_x', 'u_y', 'v_x', 'v_y', 'a_x', 'a_y']):
                a = self.tensors_list[size_block][order_1][ind, reciever]
                b = self.tensors_list[size_block][order_2][ind, reciever]

                a_interp = np.interp(t_b, t_a, a)
                errors[size_block][var].append(
                    np.max(np.abs(b - a_interp)) / np.max([np.max(np.abs(b)), np.max(np.abs(a_interp))]) * 100
                )
                print(f"error {var}: {errors[size_block][var][-1]:.2f} || delta: {np.max(np.abs(b - a_interp)):.2e} || {np.max([np.max(np.abs(b)), np.max(np.abs(a_interp))]):.2e}")
                # print(t_b[np.argmax(np.abs(b - a_interp))])
            print('-'*20)
        return errors