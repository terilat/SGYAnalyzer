import os
import numpy as np
from obspy.io.segy.segy import _read_segy
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_segy(
        path_to_exp: str,
        exp: str,
        inner_path: str = 'output'
    ) -> dict[str, np.ndarray]:
    if not os.path.isdir(path_to_exp):
        raise FileNotFoundError(f'Error path {path_to_exp}')

    pathes = {
        'Ux': os.path.join(path_to_exp, exp, inner_path, 'output_Ux.sgy'),
        'Uy': os.path.join(path_to_exp, exp, inner_path, 'output_Uy.sgy'),
        'Vx': os.path.join(path_to_exp, exp, inner_path, 'output_Vx.sgy'),
        'Vy': os.path.join(path_to_exp, exp, inner_path, 'output_Vy.sgy'),
        'Ax': os.path.join(path_to_exp, exp, inner_path, 'output_Ax.sgy'),
        'Ay': os.path.join(path_to_exp, exp, inner_path, 'output_Ay.sgy'),
        'S1': os.path.join(path_to_exp, exp, inner_path, 'output_S1.sgy'),
        'S2': os.path.join(path_to_exp, exp, inner_path, 'output_S2.sgy'),
        'P': os.path.join(path_to_exp, exp, inner_path, 'output_P.sgy')
    }
    
    tensors = {}

    for name, path in pathes.items():
        if os.path.isfile(path):
            try:
                seg = _read_segy(path).traces
                seg = np.array([trace.data for trace in seg])

                filtered_seg = []
                for i in range(len(seg)-1):
                    # if np.abs(seg[i] - seg[i+1]).sum() > np.abs(seg[i]).max() / seg[i].shape[-1] / 1e3:
                    #     filtered_seg.append(seg[i])
                    if (seg[i] != seg[i+1]).any():
                        filtered_seg.append(seg[i])
                
                if (seg[-1] != seg[-2]).any():
                    filtered_seg.append(seg[-1])

                tensors[name] = np.array(filtered_seg)[::-1]
            except Exception as e:
                print(f"Error reading {path}: {e}")
                raise e

    return tensors


def get_seysmic_tensor(path_to_exp: str, exp: str, inner_path: str='output'):
    logger.info(f"Reading {exp}")
        
    tensor = read_segy(
        path_to_exp, 
        exp=exp,
        inner_path=inner_path
    )
    tensor_shape = tensor[list(tensor.keys())[0]].shape
    logger.info(f"Tensor shape: {tensor_shape}")

    try:
        userlog_path = os.path.join(path_to_exp, exp, inner_path, 'userlog.txt')
        with open(userlog_path) as f:
            log = f.read().split('\n')
            logger.info(f"Userlog {exp} uploaded")

    except FileNotFoundError:
        logger.warning(f"Userlog {exp} not uploaded")
        time_line = np.arange(tensor_shape[-1]) / tensor_shape[-1]

    for line in log:
        match = re.search(r"Time step ([\d.eE+-]+)", line)
        if match:
            step = float(match.group(1)[:-1])
            len = tensor_shape[-1]
            time_line = np.arange(1, len + 1) * step
            logger.info(f"Time step {exp}: {step}")
            break
    
    for line in log:
        match = re.search(r"Calculation time is: ([\d.eE+-]+)", line)
        if match:
            time_evaluation = int(match.group(1))
            logger.info(f"Calculation time {exp}: {time_evaluation}")
            break

    return tensor, time_line, time_evaluation
    