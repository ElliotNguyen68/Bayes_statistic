from typing import Callable,List

import numpy as np
import math

def generate_pdf_distribution(
    pdf:Callable[[float],float],
    min_x_value:float,
    max_x_value:float,
    trial:int=10000
)->List[float]:
    x_values=np.linspace(start=min_x_value,stop=max_x_value,num=trial)

    fx=[pdf(x) for x in x_values]
    # print(fx)

    assert np.all(np.array(fx)>=0),"pdf function can't not be negative"

    fx_max=max(fx)

    gx_x=np.random.uniform(low=min_x_value,high=max_x_value,size=trial)   

    prob_gx=1/(max_x_value-min_x_value) 

    m=fx_max/prob_gx
    
    ans=[]
    for x in gx_x:
        rand=np.random.rand()
        if rand<pdf(x)/(m*prob_gx):
            ans.append(x)
    print(len(ans))
    print(np.mean(ans))
    print(np.var(ans))
    return ans

generate_pdf_distribution(lambda x: (1/math.sqrt(2*math.pi))*math.exp(-1/2*(x**2)),min_x_value=3/2,max_x_value=5/2,trial=1000)