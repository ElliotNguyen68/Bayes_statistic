from typing import Tuple

import numpy as np 


def estimate_normal_distribution_parameter_freq_1d(
    data: List[float]
)->Tuple[float,float]:
    return (np.mean(data),np.variance(data))

