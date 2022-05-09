from typing import Callable

import numpy as np


def importance_sampling(
    fx:Callable[[float],float],
    px:Callable[[float],float],
    trials: int=10000
)->float:
    """Importance sampling, used to compute expected value of a function, intergral of f(x).p(x).dx
        where p(x) is hard to sample
        Normaly, E(f(x).p(x).dx) ~ 1/N . f(xi) =A  , xi come from p(x), N is sample size, when N->oo , lim(A)=(E(f(x)))
        E(f(x)p(x)dx)=E(f(x).q(x)/q(x).p(x))=E(f(x).p(x)/q(x) . q(x)) ~  1/N . f(xi) * p(xi)/q(xi)   

    Args:
        fx (Callable[[float],float]): Funciton want to compute expected value
        px (Callable[[float],float]): Distribution of x

    Returns:
        float: E(f(x))
    """
    # We will use q(x)=Uniform(-100,100) for learning purpose, q(x)=1/200 
    # This can be replace with any other distribution easy to sample
    x_qx=np.random.uniform(-100,100,trials)
    
    E_fx=[fx(x)*px(x)/(1/200) for x in x_qx]
    
    return np.mean(E_fx)  

print(importance_sampling(fx=lambda x: 3*x-1,
                            # px=N(5,36)
                            px=lambda x:(1/(np.sqrt(2*np.pi)*6)) * np.exp(-((x-5)**2)/(2*36))) )