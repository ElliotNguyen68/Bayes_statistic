from typing import Callable,List

import numpy as np
import math

def intergration(func:Callable[[float],float],upper_bound:float,lower_bound:float,num_trials:int=10000)->float:
    """Calculate intergral of a function, using monte carlo simulation.
        We have intergral(f(x)*p(x)dx)= E(f(x))~ 1/m(f(xi)), with |x|=m
        Because a,b is uniformly distributed with pdf= 1/(b-a) which is a constant=> 1/((b-a)*intergral(h(x)dx)~ 1/m(f(xi))
        =>intergral(f(x)dx)~f(xi)(b-a)/m
    Args:
        func (Callable[[float],float]): function need to calculate 
        upper_bound (float): a in [a,b]
        lower_bound (float): b in [a,b]

    Returns:
        float: Result
    """
    xrand=np.random.uniform(low=upper_bound,high=lower_bound,size=num_trials)
    
    y_h=[func(x) for x in xrand]

    return np.sum(y_h)*(lower_bound-upper_bound)/num_trials

print(intergration(lambda x: np.sin(x),0,np.pi))
