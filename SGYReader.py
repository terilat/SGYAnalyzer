import os
import numpy as np
from obspy.io.segy.segy import _read_segy
import matplotlib.pyplot as plt
import cv2

class SGYReader:
    def __init__(self):
        pass

    def read_segy(
            self, 
            path_to_exp,
            exp
        ) -> None:
        if not os.path.isdir(path_to_exp):
            print('Error path')
        
        pathes = {
            'Ux': os.path.join(path_to_exp, exp, 'output', 'output_Ux.sgy'),
            'Uy': os.path.join(path_to_exp, exp, 'output', 'output_Uy.sgy'),
            'Vx': os.path.join(path_to_exp, exp, 'output', 'output_Vx.sgy'),
            'Vy': os.path.join(path_to_exp, exp, 'output', 'output_Vy.sgy'),
            'Ax': os.path.join(path_to_exp, exp, 'output', 'output_Ax.sgy'),
            'Ay': os.path.join(path_to_exp, exp, 'output', 'output_Ay.sgy'),
            'S1': os.path.join(path_to_exp, exp, 'output', 'output_S1.sgy'),
            'S2': os.path.join(path_to_exp, exp, 'output', 'output_S2.sgy'),
            'P': os.path.join(path_to_exp, exp, 'output', 'output_P.sgy'),
        }
        
        segs = []
        self.names = []

        for name, path in pathes.items():
            if os.path.isfile(path):
                segs.append(_read_segy(path))
                self.names.append(name)

        tensors = {}
        for seg, name in zip(segs, self.names):
            tensor = []
            for ind, trace in enumerate(seg.traces):
                tensor.append(trace.data)
            tensors[name] = np.array(tensor)

        return tensors
    
    def img_plot(self, pic_size=(1024, 512), fig_size=(16, 16)):
        fig, axs = plt.subplots(3, 2, figsize=fig_size)
        for i in range(6):
            axs[i // 2, i % 2].imshow(cv2.resize(self.tensors[i], (pic_size), interpolation=cv2.INTER_CUBIC))
            axs[i // 2, i % 2].set_title(self.names[i])
