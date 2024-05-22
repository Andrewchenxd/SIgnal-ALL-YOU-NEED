import scipy.interpolate as interpolate
import numpy as np
def chpip(data,num=2):
    t = np.linspace(0, 1, data.shape[1])
    tnew = np.linspace(0, 1, data.shape[1]*num)
    I = data[0]
    Q = data[1]
    f_pchip = interpolate.PchipInterpolator(t, I)
    Inew=f_pchip(tnew)
    f_pchip2 = interpolate.PchipInterpolator(t, Q)
    Qnew = f_pchip2(tnew)
    sgnnew=np.zeros([data.shape[0],data.shape[1]*num])
    sgnnew[0]=Inew
    sgnnew[1] = Qnew
    return sgnnew
