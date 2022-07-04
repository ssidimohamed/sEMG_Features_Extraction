from audioop import bias
from statistics import mean
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import pywt
from scipy.stats import skew, kurtosis
import antropy as ant


def features_estimation(signal, channel_name, fs, frame, step, plot=False):
    """
    Compute time, frequency and time-frequency features from signal.
    :param signal: numpy array signal.
    :param channel_name: string variable with the EMG channel name in analysis.
    :param fs: int variable with the sampling frequency used to acquire the signal
    :param frame: sliding window size
    :param step: sliding window step size
    :param plot: bolean variable to plot estimated features.

    :return: total_feature_matrix -- python Dataframe with .
    :return: features_names -- python list with

    """

    features_names = ['MEAN', 'VAR', 'SD' 'RMS', 'IEMG', 'SSI', 'MAV', 'MAV1', 'MAV2', 'MAVSLP' 'LOG', 'WL', 'ACC', 'DASDV', 'ZC', 'WAMP', 'MYOP', 'SKEW', 'KURT', 'SAMPEN', "HIST", "FR", "MNP", "TP",
                      "MNF", "MDF", "PKF", "WENT"]

    time_matrix = time_features_estimation(signal, frame, step)
    frequency_matrix = frequency_features_estimation(signal, fs, frame, step)
    time_frequency_matrix = time_frequency_features_estimation(signal, frame, step)
    total_feature_matrix = pd.DataFrame(np.column_stack((time_matrix, frequency_matrix, time_frequency_matrix)).T,
                                        index=features_names)

    print('EMG features were from channel {} extracted successfully'.format(channel_name))

    if plot:
        plot_features(signal, channel_name, fs, total_feature_matrix, step)

    return total_feature_matrix, features_names


def time_features_estimation(signal, frame, step):
    """
    Compute time features from signal using sliding window method.
    :param signal: numpy array signal.
    :param frame: sliding window size.
    :param step: sliding window step size.

    :return: time_features_matrix: narray matrix with the time features stacked by columns.
    """

    mean = []
    variance = []
    std = []
    rms = []
    iemg = []
    ssi = []
    mav = []
    mav1 = []
    mav2 = []
    mavslp = []
    log_detector = []
    wl = []
    aac = []
    dasdv = []
    zc = []
    wamp = []
    myop = []
    skw = []
    kurt = []
    sampen = []
    hist = []

    th = np.mean(signal) + 3 * np.std(signal)

    for i in range(frame, signal.size, step):
        x = signal[i - frame:i]

        mean.append(np.mean(x))
        variance.append(np.var(x))
        std.append(np.sd(x))
        rms.append(np.sqrt(np.mean(x ** 2)))
        iemg.append(np.sum(abs(x)))  # Integral
        ssi.append(np.sum(abs(x)**2))  
        mav.append(np.sum(np.absolute(x)) / frame)  # Mean Absolute Value
        mav1.append(getMAV1(signal))
        mav2.append(getMAV2(signal))
        mavslp.append(getMAVSLPk(signal))
        log_detector.append(np.exp(np.sum(np.log10(np.absolute(x))) / frame))
        wl.append(np.sum(abs(np.diff(x))))  # Wavelength
        aac.append(np.sum(abs(np.diff(x))) / frame)  # Average Amplitude Change
        dasdv.append(
            math.sqrt((1 / (frame - 1)) * np.sum((np.diff(x)) ** 2)))  # Difference absolute standard deviation value
        zc.append(zcruce(x, th))  # Zero-Crossing
        wamp.append(wilson_amplitude(x, th))  # Willison amplitude
        myop.append(myopulse(x, th))  # Myopulse percentage rate
        skw.append(skew(x, bias=False))
        kurt.append(kurtosis(x, bias=False))
        sampen.append(ant.sample_entropy(x))
        hist.append(getHIST(signal))



    time_features_matrix = np.column_stack((mean, variance, std, rms, iemg, ssi, mav, mav1, mav2, mavslp, log_detector, wl, aac, dasdv, zc, wamp, myop, skw, kurt, sampen, hist))
    return time_features_matrix


def frequency_features_estimation(signal, fs, frame, step):
    """
    Compute frequency features from signal using sliding window method.
    :param signal: numpy array signal.
    :param fs: sampling frequency of the signal.
    :param frame: sliding window size
    :param step: sliding window step size

    :return: frequency_features_matrix: narray matrix with the frequency features stacked by columns.
    """

    fr = []
    mnp = []
    tot = []
    mnf = []
    mdf = []
    pkf = []

    for i in range(frame, signal.size, step):
        x = signal[i - frame:i]
        frequency, power = spectrum(x, fs)

        fr.append(frequency_ratio(frequency, power))  # Frequency ratio
        mnp.append(np.sum(power) / len(power))  # Mean power
        tot.append(np.sum(power))  # Total power
        mnf.append(mean_freq(frequency, power))  # Mean frequency
        mdf.append(median_freq(frequency, power))  # Median frequency
        pkf.append(frequency[power.argmax()])  # Peak frequency

    frequency_features_matrix = np.column_stack((fr, mnp, tot, mnf, mdf, pkf))

    return frequency_features_matrix


def time_frequency_features_estimation(signal, frame, step):
    """
    Compute time-frequency features from signal using sliding window method.
    :param signal: numpy array signal.
    :param frame: sliding window size
    :param step: sliding window step size

    :return: h_wavelet: list
    """
    h_wavelet = []

    for i in range(frame, signal.size, step):
        x = signal[i - frame:i]

        E_a, E = wavelet_energy(x, 'db2', 4)
        E.insert(0, E_a)
        E = np.asarray(E) / 100

        h_wavelet.append(-np.sum(E * np.log2(E)))

    return h_wavelet


def wilson_amplitude(signal, th):
    x = abs(np.diff(signal))
    umbral = x >= th
    return np.sum(umbral)



def getMAV(signal):
    """ Thif functions compute the  average of EMG signal Amplitude.::
    """
    
    MAV = 1/len(signal) *  np.sum([abs(x) for x in signal])    
    return(MAV)



def getMAV1(signal):
    """ This functoin evaluate Average of EMG signal Amplitude, using the modified version n°.1.::
    """
    wIndexMin = int(0.25 * len(signal))
    wIndexMax = int(0.75 * len(signal))
    absoluteSignal = [abs(x) for x in signal]
    IEMG = 0.5 * np.sum([x for x in absoluteSignal[0:wIndexMin]]) + np.sum([x for x in absoluteSignal[wIndexMin:wIndexMax]]) + 0.5 * np.sum([x for x in absoluteSignal[wIndexMax:]])
    MAV1 = IEMG / len(signal)
    return(MAV1)


def getMAV2(signal):
    """ This functoin evaluate Average of EMG signal Amplitude, using the modified version n°.2.::
    """
    
    N = len(signal)
    wIndexMin = int(0.25 * N) #get the index at 0.25N
    wIndexMax = int(0.75 * N)#get the index at 0.75N

    temp = [] #create an empty list
    for i in range(0,wIndexMin): #case 1: i < 0.25N
        x = abs(signal[i] * (4*i/N))
        temp.append(x)
    for i in range(wIndexMin,wIndexMax+1): #case2: 0.25 <= i <= 0.75N
        x = abs(signal[i])
        temp.append(x)
    for i in range(wIndexMax+1,N): #case3; i > 0.75N
        x = abs(signal[i]) * (4*(i - N) / N)
        temp.append(x)
        
    MAV2 = np.sum(temp) / N
    return(MAV2)



def getMAVSLPk(signal, nseg):
    """ Mean Absolute value slope is a modified versions of MAV feature.
    """
    N = len(signal)
    lenK = int(N / nseg) #length of each segment to compute
    MAVSLPk = []
    for s in range(0,N,lenK):
        MAVSLPk.append(getMAV(signal[s:s+lenK]))
    return(MAVSLPk) 



def myopulse(signal, th):
    umbral = signal >= th
    return np.sum(umbral) / len(signal)


def getHIST(signal,nseg=9,th=50):
    """ Histograms is an extension version of ZC and WAMP features. 
    """
    segmentLength = int(len(signal) / nseg)
    HIST = {}
    for seg in range(0,nseg):
        HIST[seg+1] = {}
        thisSegment = signal[seg * segmentLength: (seg+1) * segmentLength]
        HIST[seg+1]["ZC"] = zcruce(thisSegment,th)
        HIST[seg+1]["WAMP"] = wilson_amplitude(thisSegment,th)
    return(HIST)



def spectrum(signal, fs):
    m = len(signal)
    n = next_power_of_2(m)
    y = np.fft.fft(signal, n)
    yh = y[0:int(n / 2 - 1)]
    fh = (fs / n) * np.arange(0, n / 2 - 1, 1)
    power = np.real(yh * np.conj(yh) / n)

    return fh, power


def frequency_ratio(frequency, power):
    power_low = power[(frequency >= 30) & (frequency <= 250)]
    power_high = power[(frequency > 250) & (frequency <= 500)]
    ULC = np.sum(power_low)
    UHC = np.sum(power_high)

    return ULC / UHC


def shannon(x):
    N = len(x)
    nb = 19
    hist, bin_edges = np.histogram(x, bins=nb)
    counts = hist / N
    nz = np.nonzero(counts)

    return np.sum(counts[nz] * np.log(counts[nz]) / np.log(2))


def zcruce(X, th):
    th = 0
    cruce = 0
    for cont in range(len(X) - 1):
        can = X[cont] * X[cont + 1]
        can2 = abs(X[cont] - X[cont + 1])
        if can < 0 and can2 > th:
            cruce = cruce + 1
    return cruce


def mean_freq(frequency, power):
    num = 0
    den = 0
    for i in range(int(len(power) / 2)):
        num += frequency[i] * power[i]
        den += power[i]

    return num / den


def median_freq(frequency, power):
    power_total = np.sum(power) / 2
    temp = 0
    tol = 0.01
    errel = 1
    i = 0

    while abs(errel) > tol:
        temp += power[i]
        errel = (power_total - temp) / power_total
        i += 1
        if errel < 0:
            errel = 0
            i -= 1

    return frequency[i]


def wavelet_energy(x, mother, nivel):
    coeffs = pywt.wavedecn(x, wavelet=mother, level=nivel)
    arr, _ = pywt.coeffs_to_array(coeffs)
    Et = np.sum(arr ** 2)
    cA = coeffs[0]
    Ea = 100 * np.sum(cA ** 2) / Et
    Ed = []

    for k in range(1, len(coeffs)):
        cD = list(coeffs[k].values())
        cD = np.asarray(cD)
        Ed.append(100 * np.sum(cD ** 2) / Et)

    return Ea, Ed


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def med_freq(f, P):
    Ptot = np.sum(P) / 2
    temp = 0
    tol = 0.01
    errel = 1
    i = 0

    while abs(errel) > tol:
        temp += P[i]
        errel = (Ptot - temp) / Ptot
        i += 1
        if errel < 0:
            errel = 0
            i -= 1

    return f[i]


def plot_features(signal, channel_name, fs, feature_matrix, step):
    """
    xxxs

    Argument:
    signal -- python numpy array representing recording of a signal.
    channel_name -- string variable with the EMG channel name in analysis.
    fs -- int variable with the sampling frequency used to acquire the signal.
    feature_matrix -- python Dataframe ...
    step -- int variable with the step size used in the sliding window method.
    """

    ts = np.arange(0, len(signal) / fs, 1 / fs)
    # for idx, f in enumerate(tfeatures.T):
    for key in feature_matrix.T:
        tf = step * (np.arange(0, len(feature_matrix.T[key]) / fs, 1 / fs))
        fig = plt.figure()

        ax = fig.add_subplot(111, label="1")
        ax2 = fig.add_subplot(111, label="2", frame_on=False)
        ax.plot(ts, signal, color="C0")
        ax.autoscale(tight=True)
        plt.title(channel_name + ": " + key)
        ax.set_xlabel("Time")
        ax.set_ylabel("mV")

        ax2.plot(tf, feature_matrix.T[key], color="red")
        ax2.yaxis.tick_right()
        ax2.autoscale(tight=True)
        ax2.set_xticks([])
        ax2.set_yticks([])
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.show()


