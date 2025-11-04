"""
Implementations of traditional echo hiding mechanisms as originally described by Gruhl et al. 
in "Echo Hiding" (http://www.fim.uni-linz.ac.at/lva/Rechtliche_Aspekte/2001SS/Stegano/leseecke/echo%20data%20hiding%20by%20d.%20gruhl%20and%20w.%20bender.pdf).

Includes three additional echo kernels beyond the original single echo kernel, specifically 
- Bipolar echo kernel (bp) : Oh et al., "Imperceptible Echo for Robust Audio Watermarking" 
- Backward-Forward echo kernel (bf) : Kim et al., "A Novel Echo-Hiding Scheme With Backward and Forward Kernels"
- Bipolar Backward-Forward echo kernel (bpbf) : Chen et al., "Highly Robust, Secure, and Perceptual-Quality Echo Hiding Schemes"
Also includes time-spread echo kernel (ts)  : Ko et al., "Time-Spread Echo Method for Digital Audio Watermarking"
"""
import os
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np

def create_filter(kernel, delay, amplitude, pn = None):
    """
    Create a filter for the specified echo kernel type, delay, and echo amplitude.

    Parameters:
        kernel : str 
            The echo kernel type. Options are 'single', 'backward_forward' (or 'bf'), 'bipolar' (or 'bp'), 'bipolar_backward_forward' (or 'bpbf'),
            and 'time_spread' (or 'ts'). If 'ts' is selected, a pseudo-noise sequence must also be provided.
        delay : int
            The delay, in samples, for the echo. For bipolar echo, they negative echo occurs 5 samples after the positive echo.
            For backward-forward echo, the backward and forward echoes are both at the specified delays.
        amplitude : float
            The amplitude of the echo.
        pn : list or np.ndarray
            The pseudo-noise sequence to use for the time spread echo kernel. If None, the kernel is not time spread.

    Returns:
        list : The filter values for the specified echo kernel.
    """
    if kernel == "single":
        filter = [1] + [0] * (delay - 1) + [amplitude]
    elif kernel == "backward_forward" or kernel == "bf":
        filter = [amplitude] + [0] * (delay - 1) + [1] + [0] * (delay - 1) + [amplitude]
    elif kernel == "bipolar" or kernel == "bp":
        filter = [1] + [0] * (delay - 1) + [amplitude] + [0]*5 + [-1*(amplitude / 2)]
    elif kernel == "bipolar_pair" or kernel == "bpp":
        filter = [1] + [0] * (delay - 1) + [-1*amplitude]
    elif kernel == "bipolar_backward_forward" or kernel == "bpbf":
        filter = [-1*amplitude] + [0] * 5 + [amplitude] + [0] * (delay - 1) +  [1]  + [0] * (delay - 1) + [amplitude] + [0] * 5 + [-1*amplitude]
    elif kernel == "time_spread" or kernel == "ts":
        if pn is None:
            raise ValueError("Pseudo-noise sequence required for time spread echo kernel.")
        filter = [1] + [0] * (delay - 1) + [amplitude*pn[j] for j in range(len(pn))]
    else:
        raise ValueError(f"Invalid kernel type: {kernel}")
    return filter


def create_filter_bank(kernel, delays, amplitude, pn = None):
    """
    Create a bank of filters -- one filter per delay -- for the specified echo kernel type, delays, and echo amplitude. 

    Parameters:
        kernel : str
            The echo kernel type. Options are 'single', 'backward_forward' (or 'bf'), 'bipolar' (or 'bp'), 'bipolar_backward_forward' (or 'bpbf'),
            and 'time_spread' (or 'ts'). If 'ts' is selected, a pseudo-noise sequence must also be provided.
        delays : list
            The list of delays, in samples, for the echo. 
        amplitude : float
            The amplitude of the echo.
        pn : list or np.ndarray
            The pseudo-noise sequence to use for the time spread echo kernel. If None, the kernel is not time spread.

    Returns:
        list : A list of filters, one for each delay in the delays list.
    """
    filters = [] 
    for i in range(len(delays)):
        filter = create_filter(kernel, delays[i], amplitude, pn = pn)  
        filters.append(filter)
    return filters
    

def encode(audio, symbols, amplitude, delays, win_size, kernel, pn = None, filters = None, hanning_factor = 4):
    """
    Given an audio signal, a set of delays serving as the symbol "alphabet," and other echo parameters
    encode the provided symbols onto the audio signal.

    Parameters:
        audio : np.ndarray
            The raw audio signal, mono-channel.
        symbols : list
            The list of symbols to encode onto the audio signal. Each should be an index into the delays list.
        amplitude : float
            The amplitude of the echo.
        delays : list
            The list of delay options for encoding.
        win_size : int
            The window size for encoding. Each window encodes one sample by independently lagging the portion of 
            the audio signal within it.
        kernel : str
            The echo kernel type. 
            Options are 'single', 'backward_forward' (or 'bf'), 'bipolar' (or 'bp'), and 'bipolar_backward_forward' (or 'bpbf'), 
            or 'time_spread' (or 'ts'). If 'ts' is selected, a pseudo-noise sequence must be provided (see below).
        pn : list or np.ndarray
            The pseudo-noise sequence to use for the time spread echo kernel. If None, the kernel is not time spread.
        filters : list
            Optionally, a premade filter bank can be provided. If None, the filters are created from the kernel and delays, at
            each invocation of encode(). Note: the filters list is assumed to contain elements in order matching that of the inputted delays.
        hanning_factor : int
            The factor by which the window size is divided to determing the Hanning window size. The Hanning window is used to smooth the square wave symbols.
            A larger factor results in a smaller Hanning window, which reduces the amount of smoothing but benefits encoding robustness.
        
    Returns:
        np.ndarray : The encoded audio signal.    
    """
    audio_dtype = audio.dtype

    if filters is None:
        filters = []
        for i in range(len(delays)):
            filter = create_filter(kernel, delays[i], amplitude, pn = pn)  
            filters.append(filter) 
  
    filtered_signals = []
    for i in range(len(filters)):
        filtered_signal  = signal.fftconvolve(audio, filters[i], mode = "same") #signal.lfilter(filters[i], 1, audio)
        filtered_signals.append(filtered_signal)

    hanning = np.hanning(win_size // hanning_factor) 
    symbol_square_waves = [] 
    for i in range(len(delays)):
        symbol_bool = [symbols[j] == i for j in range(len(symbols))]
        symbol_square_wave = np.repeat(symbol_bool, win_size)
        symbol_square_wave = symbol_square_wave
        symbol_square_waves.append(symbol_square_wave)
    
    mixers = []
    for i in range(len(delays)):
        mixer = signal.fftconvolve(symbol_square_waves[i], hanning, mode = "same") # signal.lfilter(hanning, 1, symbol_square_waves[i])
        if mixer.max() - mixer.min() != 0:
            mixer = (mixer - mixer.min()) / (mixer.max() - mixer.min())
        elif mixer.max() == 0:
            mixer = np.zeros(len(mixer))
        elif mixer.max() == 1:
            mixer = np.ones(len(mixer))
        mixers.append(mixer)
    

    final = np.zeros(len(audio))
    for i in range(len(delays)):
        mixed = filtered_signals[i] * mixers[i]
        final += mixed

    final = final.astype(audio_dtype) # preserve original audio dtype

    return final



def butter_highpass(cutoff, fs, order=5):
    """
    Create a Butterworth high-pass filter.

    Parameters:
        cutoff : float
            The cutoff frequency for the high-pass filter.
        fs : int
            The sampling rate of the signal.
        order : int
            The order of the Butterworth filter.
    
    Returns:
        tuple : The filter coefficients.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    """
    Apply a Butterworth high-pass filter to a signal.

    Parameters: 
        data : np.ndarray
            The signal to filter.
        cutoff : float
            The cutoff frequency for the high-pass filter.
        fs : int
            The sampling rate of the signal.
        order : int
            The order of the Butterworth filter.
    
    Returns:
        np.ndarray : The filtered signal.
    """
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def get_cepstrum(signal):
    """
    Get the cepstrum of a signal.

    Parameters:
        signal : np.ndarray
            The signal to compute the cepstrum of.
    
    Returns:
        np.ndarray : The cepstrum of the signal.
    """
    fft = np.fft.fft(signal)
    sqr_log_fft = np.log(np.abs(fft) + 0.00001)
    cepstrum = np.fft.ifft(sqr_log_fft).real
    return cepstrum

def get_autocepstrum_gruhl(signal):
    """
    Get the autocepstrum of a signal as defined by Gruhl et al. (incorrectly)
    as the autocorrelation of the cepstrum of a singal

    Parameters:
        signal : np.ndarray
            The signal to compute the autocepstrum of.
    
    Returns:
        np.ndarray : The autocepstrum of the signal.
    """
    fft = np.fft.fft(signal)
    sqr_log_fft = np.log(np.abs(fft) + 0.00001)
    cepstrum = np.fft.ifft(sqr_log_fft)
    cepstrum = cepstrum[:len(cepstrum)//2]
    autocorr = np.correlate(cepstrum, np.conj(cepstrum), mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = np.abs(autocorr)
    return autocorr

def get_autocepstrum(signal):
    """
    Get autocepstrum of a signal, defined in multiple sources as the cepstrum of the autocorrelation of a signal.
    e.g., https://stackoverflow.com/questions/14353869/autocepstrum-accelerate-framework, Wikipedia, etc.

    Parameters:
        signal : np.ndarray
            The signal to compute the autocepstrum of.
    
    Returns:
        np.ndarray : The autocepstrum of the signal.
    """
    autocorr = np.correlate(signal, signal, mode='full')
    cep_autocorr = np.fft.ifft(np.log(np.abs(np.fft.fft(autocorr)) + 0.00001)).real
    return cep_autocorr


def decode(audio, delays, win_size, sampling_rate, pn = None, cutoff_freq = None, plot = False, save_plot_path = None, gt = None):
    """
    Decode an encoded audio signal produced by encode() above. I implement decoding based on the cepstrum
    of signals, as well as the decoding based on the autocepstrum (cepstrum of the autocorrelation of the signal).

    Parameters:
        audio : np.ndarray
            The encoded audio signal.
        delays : list
            The list of delay options used in encoding.
        win_size : int
            The window size used in encoding.
        sampling_rate : int
            The sampling rate of the audio signal.
        pn : list
            The pseudo-noise sequence used in encoding, if any. If provided, assumes the time spread echo kernel was used
            for encoding.
        cutoff_freq : float
            Optionally, apply a high-pass filter to the audio signal before decoding, as some 
            initial experiments on speech audio indicate this can reduce natural BER.
        plot : bool
            Optionally, plot the decoding process, displaying results for each window.
        gt : list
            Optionally, provide the ground truth symbols for each window, to compare against the decoded symbols.
            This is used by the plots for a nice demo but nothing else in the decoding process.
    
    Returns:
        tuple of lists : The predicted symbols using cepstrum and autocepstrum decoding methods, respectively.
    """
    num_wins = len(audio) // win_size
    pred_symbols = []
    pred_symbols_autocepstrum = []

    if cutoff_freq is not None: # high-pass filter the entire audio
        audio = butter_highpass_filter(audio, cutoff_freq, sampling_rate)

    for i in range(num_wins):
        win = audio[i * win_size : (i + 1) * win_size]

        # cepstrum peak decoding
        cepstrum = get_cepstrum(win)
        if pn is not None:
            cepstrum = np.correlate(cepstrum, pn) # unspread the cepstrum 
        cep_vals = cepstrum[delays]
        max_val = np.argmax(cep_vals)
        max_cep_del = delays[max_val]
        pred_symbols.append(max_val)

        # autocepstrum peak decoding
        autocepstrum = get_autocepstrum(win)
        if pn is not None:
            autocepstrum = np.correlate(autocepstrum, pn) # unspread the autocepstrum
        autocepstrum_min = np.min(autocepstrum[3:-3])
        autocepstrum_max = np.max(autocepstrum[3:-3])
        autocepstrum_vals  = autocepstrum[delays]
        max_autocepstrum_val  = np.argmax(autocepstrum_vals )
        max_autocepstrum_del  = delays[max_autocepstrum_val ]
        pred_symbols_autocepstrum.append(max_autocepstrum_val )

        if plot or save_plot_path is not None:
            fig, axes = plt.subplots(3, 1, figsize = (12, 5), tight_layout = True)
            axes[0].plot(win)   
            
            cep_min = np.min(cepstrum[3:-3])
            cep_max = np.max(cepstrum[3:-3])
            axes[1].plot(cepstrum)
            axes[1].vlines(delays, cep_min, cep_max, linestyles = "dashed", color = 'gray', alpha = 0.5)
            if gt is not None:
                axes[1].vlines(delays[gt[i]], cep_min, cep_max, linestyles = "dashed", color = 'green', alpha = 0.5)
            axes[1].set_title("Cepstrum")
            axes[1].set_xlim(1, max(delays) + 10)
            axes[1].set_ylim(cep_min, cep_max)

           
            axes[2].plot(autocepstrum)
            axes[2].set_title("Autocorrelation of Cepstrum")
            axes[2].vlines(delays, cep_min, cep_max, linestyles = "dashed", color = 'gray', alpha = 0.5)
            if gt is not None:
                axes[2].vlines(delays[gt[i]], cep_min, cep_max, linestyles = "dashed", color = 'green', alpha = 0.5)
            axes[2].set_xlim(1, max(delays) + 10)
            axes[2].set_ylim(autocepstrum_min, autocepstrum_max)
            
            if gt is not None:
                title = f"VWindow {i}. Cep Decoded {max_cep_del}. Autocep Decoded {max_autocepstrum_del}. GT {delays[gt[i]]}"
            else:
                title = f"Window {i}. Cep Decoded {max_cep_del}. Autocep Decoded {max_autocepstrum_del}"
            plt.suptitle(title)

            if plot:
                plt.show()
            if save_plot_path is not None:
                if not os.path.exists(save_plot_path):
                    os.makedirs(save_plot_path)
                plt.savefig(f"{save_plot_path}/cepstrum_win{i}.png")
            plt.close()
      
    return pred_symbols, pred_symbols_autocepstrum

def bits_to_symbols(bits, alphabet_size):
    """
    Convert a list of bits to a list of symbols based on the provided list of symbols
    """
    """
    Parameters:
        bits : list of int
            The list of bits to convert to symbols.
        alphabet_size : int
            The size of the alphabet, which determines how many bits correspond to each symbol.
    Returns:
        list : The list of symbols corresponding to the input bits.
    """
    # each group of n bits corresponds to one symbol. map each group of n bits (str) to an int
    n = int(np.log2(alphabet_size))  # number of bits per symbol
    # TODO: add padding of zero to bits if len(bits) is not a multiple of n, so that the last group of bits is not lost.
    symbols = [int("".join(map(str, bits[i:i + n])), 2) for i in range(0, len(bits), n) if len(bits[i:i + n]) == n]
    return symbols

def symbols_to_bits(symbols, alphabet_size):
    """
    Convert a list of symbols back to a list of bits based on the provided alphabet size.

    Parameters:
        symbols : list of int
            The list of symbols to convert to bits.
        alphabet_size : int
            The size of the alphabet, which determines how many bits correspond to each symbol.

    Returns:
        list : The list of bits corresponding to the input symbols.
    """
    n = int(np.log2(alphabet_size))  # number of bits per symbol
    assert max(symbols) < 2**n, f"All symbols must be leq 2**n, where n=log2(alphabet_size) ({2**n} in this case), but found a symbol with value {max(symbols)} in input"
    bits = []
    for symbol in symbols:
        # Convert each symbol to its binary representation and pad with zeros
        # print(symbol, n,  alphabet_size)
        bin_repr = np.binary_repr(symbol, width=n)
        bits.extend(map(int, bin_repr))
    return bits


def encode_with_sync(audio, payload_symbols, amplitude, payload_delays, payload_win_size, kernel,
                    preamble_delays, preamble_frequency, preamble_win_size,
                    pn = None, payload_filters = None, preamble_filters = None, hanning_factor = 4):
    """
    Given an audio signal, a set of delays serving as the symbol "alphabet," and other echo parameters
    encode the provided symbols onto the audio signal.

    Parameters:
        audio : np.ndarray
            The raw audio signal, mono-channel.
        payload_symbols : list
            The list of data symbols to encode onto the audio signal. Each should be an index into the payload_delays list.
        amplitude : float
            The amplitude of the echo.
        payload_delays : list
            The list of delay options for encoding of the payload symbols.
        payload_win_size : int
            The window size for encoding symbols of the payload. Each window encodes one sample by independently lagging the portion of
            the audio signal within it.
        kernel : str
            The echo kernel type. 
            Options are 'single', 'backward_forward' (or 'bf'), 'bipolar' (or 'bp'), and 'bipolar_backward_forward' (or 'bpbf'), 
            or 'time_spread' (or 'ts'). If 'ts' is selected, a pseudo-noise sequence must be provided (see below).
        pn : list or np.ndarray
            The pseudo-noise sequence to use for the time spread echo kernel. If None, the kernel is not time spread.
        filters : list
            Optionally, a premade filter bank can be provided. If None, the filters are created from the kernel and delays, at
            each invocation of encode(). Note: the filters list is assumed to contain elements in order matching that of the inputted delays.
        hanning_factor : int
            The factor by which the window size is divided to determing the Hanning window size. The Hanning window is used to smooth the square wave symbols.
            A larger factor results in a smaller Hanning window, which reduces the amount of smoothing but benefits encoding robustness.
        
    Returns:
        np.ndarray : The encoded audio signal.    
    """
    audio_dtype = audio.dtype

    # for symbols
    if payload_filters is None:
        payload_filters = []
        for i in range(len(payload_delays)):
            filter = create_filter(kernel, payload_delays[i], amplitude, pn = pn)  
            payload_filters.append(filter) 
    payload_filtered_signals = [] # create filtered signals for each delay
    for i in range(len(payload_filters)):
        payload_filtered_signal  = signal.fftconvolve(audio, payload_filters[i], mode = "same") 
        payload_filtered_signals.append(payload_filtered_signal)

    # for sync
    if preamble_filters is None:
        preamble_filters = []
        for i in range(len(preamble_delays)):
            filter = create_filter(kernel, preamble_delays[i], amplitude, pn = pn)  
            preamble_filters.append(filter)
    preamble_filtered_signals = [] # create filtered signals for each preamble delay
    for i in range(len(preamble_filters)):
        filtered_signal  = signal.fftconvolve(audio, preamble_filters[i], mode = "same") 
        preamble_filtered_signals.append(filtered_signal)

    # create mixers for symbols, making sure to interject zeros for preambles of sync symbols
    hanning = np.hanning(payload_win_size // hanning_factor) 
    preamble_size = preamble_win_size * len(preamble_delays)
    data_portion_size = payload_win_size * preamble_frequency
    frame_size = preamble_size + payload_win_size * preamble_frequency # total number of samples per frame (preamble + data windows)
    num_frames = len(audio) // frame_size
    payload_symbol_square_waves = [] 
    is_preamble_square_wave = np.zeros(len(audio), dtype = int) # only for debugging
    for i in range(len(payload_delays)):
        symbol_bool = [payload_symbols[j] == i for j in range(len(payload_symbols))] # symbol-specific boolean array
        symbol_square_wave = np.repeat(symbol_bool, payload_win_size)
        # interject symbol_square_wave with array of zeros of length preamble_size*len(preamble_delays) every preamble_frequency windows
        symbol_square_wave_with_sync = []
        for j in range(num_frames):
            data_portion = symbol_square_wave[j*data_portion_size:(j+1)*data_portion_size]
            symbol_square_wave_with_sync.extend([0] * preamble_size)
            symbol_square_wave_with_sync.extend(data_portion)
            is_preamble_square_wave[j*frame_size:j*frame_size + preamble_size] = 1 # only for debuggigng
        payload_symbol_square_waves.append(symbol_square_wave_with_sync)
        # plt.figure(figsize=(12, 4))
        # plt.plot(symbol_square_wave_with_sync)
        # plt.plot(is_preamble_square_wave, color = 'red', alpha = 0.5)
        # plt.title(f"Delay {i}. Symbols: {symbols[:20]}")
        # plt.xlim(0, frame_size * 2)
        # plt.show()
    
    # build preamble square wave for each preamble delay
    preamble_delay_square_waves = []
    for i in range(len(preamble_delays)):
        preamble_delay_bool = [i == j for j in range(len(preamble_delays))]
        preamble_delay_square_wave = np.repeat(preamble_delay_bool, preamble_win_size)
        preamble_delay_square_wave = np.concatenate([preamble_delay_square_wave, [0] * data_portion_size])
        # repeat the preamble square wave for each frame
        preamble_delay_square_wave = np.tile(preamble_delay_square_wave, num_frames)
        preamble_delay_square_waves.append(preamble_delay_square_wave)
        # plt.figure(figsize=(12, 4))
        # plt.plot(preamble_delay_square_wave)
        # plt.plot(is_preamble_square_wave, color = 'red', alpha = 0.5)
        # plt.title(f"Preamble Delay {preamble_delays[i]}")
        # plt.xlim(0, frame_size * 2)
        # plt.show()

    payload_symbol_mixers = []
    for i in range(len(payload_delays)):
        mixer = signal.fftconvolve(payload_symbol_square_waves[i], hanning, mode = "same") # signal.lfilter(hanning, 1, symbol_square_waves[i])
        if mixer.max() - mixer.min() != 0:
            mixer = (mixer - mixer.min()) / (mixer.max() - mixer.min())
        elif mixer.max() == 0:
            mixer = np.zeros(len(mixer))
        elif mixer.max() == 1:
            mixer = np.ones(len(mixer))
        payload_symbol_mixers.append(mixer)
    preamble_delay_mixers = []
    for i in range(len(preamble_delays)):
        mixer = signal.fftconvolve(preamble_delay_square_waves[i], hanning, mode = "same") # signal.lfilter(hanning, 1, symbol_square_waves[i])
        if mixer.max() - mixer.min() != 0:
            mixer = (mixer - mixer.min()) / (mixer.max() - mixer.min())
        elif mixer.max() == 0:
            mixer = np.zeros(len(mixer))
        elif mixer.max() == 1:
            mixer = np.ones(len(mixer))
        preamble_delay_mixers.append(mixer)

    final = np.zeros(len(audio))
    for i in range(len(payload_delays)):
        mixed = payload_filtered_signals[i] * payload_symbol_mixers[i]
        final += mixed
    for i in range(len(preamble_delays)):
        mixed = preamble_filtered_signals[i] * preamble_delay_mixers[i]
        final += mixed

    final = final.astype(audio_dtype) # preserve original audio dtype

    return final

def decode_with_sync(audio, delays, payload_win_size, preamble_delays, preamble_frequency, preamble_win_size,sampling_rate, pn = None, cutoff_freq = None, plot = False, save_plot_path = None, gt = None):
    """
    Decode an encoded audio signal produced by encode() above. I implement decoding based on the cepstrum
    of signals, as well as the decoding based on the autocepstrum (cepstrum of the autocorrelation of the signal).

    Parameters:
        audio : np.ndarray
            The encoded audio signal.
        delays : list
            The list of delay options used in encoding.
        win_size : int
            The window size used in encoding.
        sampling_rate : int
            The sampling rate of the audio signal.
        pn : list
            The pseudo-noise sequence used in encoding, if any. If provided, assumes the time spread echo kernel was used
            for encoding.
        cutoff_freq : float
            Optionally, apply a high-pass filter to the audio signal before decoding, as some 
            initial experiments on speech audio indicate this can reduce natural BER.
        plot : bool
            Optionally, plot the decoding process, displaying results for each window.
        gt : list
            Optionally, provide the ground truth symbols for each window, to compare against the decoded symbols.
            This is used by the plots for a nice demo but nothing else in the decoding process.
    
    Returns:
        tuple of lists : The predicted symbols using cepstrum and autocepstrum decoding methods, respectively.
    """
    preamble_size = preamble_win_size * len(preamble_delays)
    frame_size = preamble_size + payload_win_size * preamble_frequency
    num_frames = len(audio) // frame_size
    pred_symbols = []
    pred_symbols_autocepstrum = []

    if cutoff_freq is not None: # high-pass filter the entire audio
        audio = butter_highpass_filter(audio, cutoff_freq, sampling_rate)

    for i in range(num_frames):
        frame = audio[i * frame_size : (i + 1) * frame_size]
        
        # for p in range(len(preamble_delays)): # optionally verify the preamble first
        #     preamble_win = frame[p*preamble_win_size : (p+1)*preamble_win_size]
        #     preamble_cepstrum = get_cepstrum(preamble_win)
        #     fig, axes = plt.subplots(2, 1, figsize = (12, 5), tight_layout = True)
        #     axes[0].plot(preamble_win)
        #     cep_min = np.min(preamble_cepstrum[3:-3])
        #     cep_max = np.max(preamble_cepstrum[3:-3])
        #     axes[1].plot(preamble_cepstrum)
        #     axes[1].vlines(preamble_delays[p], cep_min, cep_max, linestyles = "dashed", color = 'gray', alpha = 0.5)
        #     axes[1].set_title(f"Cepstrum for Preamble Delay #{p} (Delay {preamble_delays[p]})")
        #     axes[1].set_xlim(1, max(delays) + 10)
        #     axes[1].set_ylim(cep_min, cep_max)
        #     plt.show()

        for j in range(preamble_frequency):
            win = frame[preamble_size + j * payload_win_size : preamble_size + (j + 1) * payload_win_size]

            # cepstrum peak decoding
            cepstrum = get_cepstrum(win)
            if pn is not None:
                cepstrum = np.correlate(cepstrum, pn) # unspread the cepstrum 
            cep_vals = cepstrum[delays]
            max_val = np.argmax(cep_vals)
            max_cep_del = delays[max_val]
            pred_symbols.append(max_val)

            # autocepstrum peak decoding
            autocepstrum = get_autocepstrum(win)
            if pn is not None:
                autocepstrum = np.correlate(autocepstrum, pn) # unspread the autocepstrum
            autocepstrum_min = np.min(autocepstrum[3:-3])
            autocepstrum_max = np.max(autocepstrum[3:-3])
            autocepstrum_vals  = autocepstrum[delays]
            max_autocepstrum_val  = np.argmax(autocepstrum_vals )
            max_autocepstrum_del  = delays[max_autocepstrum_val ]
            pred_symbols_autocepstrum.append(max_autocepstrum_val )

            if plot or save_plot_path is not None:
                sym_num = i * preamble_frequency + j

                fig, axes = plt.subplots(3, 1, figsize = (12, 5), tight_layout = True)
                axes[0].plot(win)   
        
                cep_min = np.min(cepstrum[3:-3])
                cep_max = np.max(cepstrum[3:-3])
                axes[1].plot(cepstrum)
                axes[1].vlines(delays, cep_min, cep_max, linestyles = "dashed", color = 'gray', alpha = 0.5)
                if gt is not None:
                    axes[1].vlines(delays[gt[sym_num]], cep_min, cep_max, linestyles = "dashed", color = 'green', alpha = 0.5)
                axes[1].set_title("Cepstrum")
                axes[1].set_xlim(1, max(delays) + 10)
                axes[1].set_ylim(cep_min, cep_max)

            
                axes[2].plot(autocepstrum)
                axes[2].set_title("Autocorrelation of Cepstrum")
                axes[2].vlines(delays, cep_min, cep_max, linestyles = "dashed", color = 'gray', alpha = 0.5)
                if gt is not None:
                    axes[2].vlines(delays[gt[sym_num]], cep_min, cep_max, linestyles = "dashed", color = 'green', alpha = 0.5)
                axes[2].set_xlim(1, max(delays) + 10)
                axes[2].set_ylim(autocepstrum_min, autocepstrum_max)
                
                if gt is not None:
                    title = f"Frame {i} Window {j}. Cep Decoded {max_cep_del}. Autocep Decoded {max_autocepstrum_del}. GT {delays[gt[sym_num]]}"
                else:
                    title = f"Frame {i} Window {j}. Cep Decoded {max_cep_del}. Autocep Decoded {max_autocepstrum_del}"
                plt.suptitle(title)

                if plot:
                    plt.show()
                if save_plot_path is not None:
                    if not os.path.exists(save_plot_path):
                        os.makedirs(save_plot_path)
                    plt.savefig(f"{save_plot_path}/cepstrum_win{i}.png")
                plt.close()
      
    return pred_symbols, pred_symbols_autocepstrum



###############################
## USAGE DEMO: PRELIMINARIES ##
# #############################
# import soundfile
# set the general parameters for encoding/decoding
# delays = [i*2 for i in range(20, 52)] # recommended delay range for 44.1/48kHz audio is 40-120 for best imperceptibility. Adjust accordingly for other sample rates.
# amplitude = 0.4 # recommended ~0.4 for best imperceptibility
# win_size = 3000 # recommended ~3000 samples for  44.1/48kHz audio for reasonable imperceptibility-robustness tradeoff
# hanning_factor = 4
# kernel = "bp"
# if kernel == "ts":
#     pn_size = 4000
#     assert win_size > pn_size
#     assert amplitude < 0.1 # weird things happen if amplitude is too high
#     pn = np.random.choice([-1, 1], size = pn_size) # generate a random pseudo-noise sequence
# else:
#     pn = None
# cutoff_freq = 1000 # for optional high-pass filtering the audio before decoding. Set to None to skip filtering.

# # sync/preamble parameters
# preamble_frequency = 10 # insert a sync preamble every n windows
# preamble_win_size = 6000 # window size for each sync symbol
# preamble_delays = [40, 50, 100, 30]
# preamble_size = preamble_win_size * len(preamble_delays)
# frame_size = preamble_size + win_size * preamble_frequency # total number of samples per frame (preamble + data windows)

#############
# SYNC DEMO #
#############
# load the audio file
# audio, SR = soundfile.read("/Users/hadleigh/ears_reverb_p001/emo_serenity_freeform.wav") # load a mono audio file

# # determine the number of windows and trim the audio to fit
# print("Frame size (samples): ", frame_size)
# print("Number of data symbol samples per frame: ", win_size * preamble_frequency)
# print("Preamble size (samples): ", preamble_size)
# num_frames = len(audio) // frame_size
# num_sym_wins = num_frames * preamble_frequency
# print(f"Audio length: {len(audio)} samples. Number of frames: {num_frames}. Number of symbol windows: {num_sym_wins}")
# audio = audio[:num_frames * frame_size]

# # create some dummy data to encode as example
# symbols = np.random.randint(0, len(delays), size = num_sym_wins) # directly generate random symbols
# # or generate random bits and convert to symbols
# # num_bits = int(np.log2(len(delays))) * num_sym_wins # number of bits per symbol
# # print(f"Generating {num_bits} random bits for {num_sym_wins} windows")
# # bits = np.random.randint(0, 2, size = num_bits) # generate random bits
# # symbols = bits_to_symbols(bits, len(delays)) # convert bits to symbols

# # encode the audio and save it
# encoded = encode_with_sync(audio, symbols, amplitude, delays, win_size, kernel,
#                             preamble_delays, preamble_frequency, preamble_win_size,
#                             pn = pn)
# soundfile.write("encoded_audio.wav", encoded, SR)
# pred_symbols, _ = decode_with_sync(encoded, delays, win_size, preamble_delays, preamble_frequency, preamble_win_size, SR, pn = pn, gt = symbols, plot = False, cutoff_freq = cutoff_freq)
# num_symbol_errs = np.sum(np.array(pred_symbols) != np.array(symbols))
# print(f"Number of symbol errors: {num_symbol_errs}/{len(symbols)}")


################
# NO SYNC DEMO #
################
## load the audio file
# audio, SR = soundfile.read("/Users/hadleigh/ears_reverb_p001/sentences_01_fast.wav") # load a mono audio file
## for version without sync:
## determine the number of windows and trim the audio to fit
# num_sym_wins = len(audio) // win_size
# audio = audio[:num_sym_wins * win_size]
## create some dummy data to encode as example
# symbols = np.random.randint(0, len(delays), size = num_sym_wins) # directly generate random symbols
## or generate random bits and convert to symbols
# num_bits = int(np.log2(len(delays))) * num_sym_wins # number of bits per symbol
# print(f"Generating {num_bits} random bits for {num_sym_wins} windows")
# bits = np.random.randint(0, 2, size = num_bits) # generate random bits
# symbols = bits_to_symbols(bits, len(delays)) # convert bits to symbols
# # encode the audio and save it
# encoded = encode(audio, symbols, amplitude, delays, win_size, kernel, pn = pn)
# soundfile.write("encoded_audio.wav", encoded, SR)
# # decode the audio and determine the bit error rate (BER) and symbol error rate
# pred_symbols, _  = decode(encoded, delays, win_size, SR, pn = pn, gt = symbols, plot = False, cutoff_freq = cutoff_freq)
# pred_bits = symbols_to_bits(pred_symbols, len(delays))
# ber = np.sum(np.array(pred_bits) != np.array(bits)) / len(pred_bits)
# print(f"BER: {ber:.4f} for {len(pred_bits)} bits")
# num_symbol_errs = np.sum(np.array(pred_symbols) != np.array(symbols))
# print(f"Number of symbol errors: {num_symbol_errs}/{len(symbols)}")




