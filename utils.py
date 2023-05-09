import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def get_psd_feature(
    dataframe,
    freq_type: str,
    fs: int = 256,
    len_drop: int = 7680,
    len_keep: int = 46080,
    plot_psd: bool = False,
    save_psd: bool = False,
    channel_drop: list = None,
    print_summary: bool = False,
):
    
    # 1. Only get the signal from the dataframe
    df_signal = dataframe[['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']]
    
    # Drop channels if `channel_drop` is not None
    if channel_drop is not None:
        df_signal = df_signal.drop(columns=channel_drop)
    
    # 2. Crop the signal
    df_signal = df_signal.iloc[len_drop:]
    df_signal = df_signal.iloc[:len_keep]
    
    # 3. Transpose the dataframe
    df_signal = df_signal.T
    
    # 4. Transform DF into MNE object
    info = mne.create_info(
        ch_names = df_signal.shape[0],
        sfreq = fs,
        ch_types = 'eeg',
        verbose= False,
    )
    
    raw = mne.io.RawArray(
        data = df_signal,
        info = info,
        verbose = False,
    )
    
    # 5. Get the PSD for RAW Signal
    psd_raw, freqs = mne.time_frequency.psd_array_multitaper(
        x = raw.get_data(),
        sfreq = fs,
        verbose = False,
    )
    
    # 6. Get the PSD for Selected Bands
    freq_bands = {'delta': (0.5, 4),
              'theta': (4, 8),
              'alpha': (8, 13),
              'beta': (13, 30),
              'gamma': (30, 50)}
    
    # obtain frequency range based on the parameter passed to `freq_type`
    freq_range = freq_bands[freq_type]
    raw_copy = raw.copy()
    raw_copy.filter(freq_range[0], freq_range[1], fir_design = 'firwin', verbose=False)
    
    psd_filered, freqs_filtered = mne.time_frequency.psd_array_multitaper(
        x = raw_copy.get_data(),
        sfreq = fs,
        verbose = False,
    )
    
    # 7. Count the features
    sum_raw = np.sum(psd_raw)
    avg_raw = np.average(psd_raw)
    
    sum_filtered = np.sum(psd_filered)
    avg_filtered = np.average(psd_filered)
    
    rel_pow = sum_filtered / sum_raw
    
    # 8. Return
    
    output = {
        'sum_raw': sum_raw,
        'avg_raw': avg_raw,
        'sum_filtered': sum_filtered,
        'avg_filtered': avg_filtered,
        'rel_pow': rel_pow,
    }
    
    return output
    
        