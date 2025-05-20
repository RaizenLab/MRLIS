import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.constants as con
matplotlib.use('TkAgg')

def averageV(T,m):
    vav = np.sqrt((8*con.k*T)/(m*con.pi))
    return vav

def doppler(w0,v,thet):
    dFreq  = w0*(v/con.c)*np.sin(thet)
    return dFreq

def absorptionSignal(w,w0,dw,dShift):
    absPlus = (dw/2)**2 + (w-w0+dShift)**2
    absMin = (dw/2)**2 + (w-w0-dShift)**2
    res = (4*dShift*(w0-w))/(absPlus*absMin)
    return res

def main():
    thetaMax = 0.0195
    waveL = 460.73330e-9
    freq = con.c/waveL
    ovenTemp = 530+273.15
    mSr = 1.4549642e-25
    einsteinA = 2.01e8
    linewidth = (einsteinA)/(2*con.pi)

    vOven = averageV(ovenTemp, mSr)
    shift = doppler(freq, vOven, thetaMax)

    minFreq = freq - 0.5e9
    maxFreq = freq + 0.5e9
    frequencies = np.linspace(minFreq, maxFreq, num=1000, endpoint=True)
    absS = np.array(absorptionSignal(frequencies,freq, linewidth, shift))
    plt.plot(frequencies,absS)
    plt.show()

    print("The central resonance frequency is " + str(freq*1e-9) + "GHz")
    print("The Natural Linewidth is " + str(linewidth*1e-6)+ "MHz")
    print("The Doppler Shift is " + str(shift*1e-6) + "MHz")
    


if __name__ == "__main__":
    main()