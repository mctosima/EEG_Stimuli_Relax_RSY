import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import stats
import csv

def get_psd_feature(
    dataframe,
    freq_type: str,
    fs: int = 256,
    len_drop: int = 7680,
    len_keep: int = 46080,
    plot_psd: bool = False,
    return_psd: bool = False,
    channel_drop: list = None,
    select_channels: list = None,
):
    
    # 1. Only get the signal from the dataframe
    df_signal = dataframe[['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']]
    
    # Drop channels if `channel_drop` is not None
    if channel_drop is not None:
        df_signal = df_signal.drop(columns=channel_drop)
        
    # Select certain channels based on `select_channels`
    if select_channels is not None:
        df_signal = df_signal[select_channels]
    
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
    
    if plot_psd:
        # Plot the PSD
        plt.figure(figsize=(10, 5))
        plt.plot(freqs_filtered, psd_filered.mean(0), color='r')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density (dB)')
        plt.title('PSD of filtered signal')
        xlim = (freq_range[0]-2, freq_range[1]+2)
        plt.xlim(xlim)        
        plt.show()
        
    if return_psd:
        output['psd_raw'] = psd_raw
        output['psd_raw_freqs'] = freqs
        output['psd_filtered'] = psd_filered
        output['psd_filtered_freqs'] = freqs_filtered
    
    return output

def load_mnedf(edf_path):
    signal_df = mne.io.read_raw_edf(edf_path, preload=True, verbose=False).to_data_frame()
    return signal_df

def get_ttest(csv_path, noise, task, freq, feature, savelog=False):
    # 1. read csv file
    df = pd.read_csv(csv_path)

    # 2. get data
    df_silent = df[(df['noise_type'] == 'silent') & (df['task'] == task) & (df['freq_type'] == freq)]
    df_noise = df[(df['noise_type'] == noise) & (df['task'] == task) & (df['freq_type'] == freq)]

    # 3. get feature
    group_silent = df_silent[feature]
    group_noise = df_noise[feature]

    # 4. Levene's test
    t_levene, p_levene = stats.levene(group_silent, group_noise)

    # 5. t-test
    if p_levene < 0.05:
        ttest_val, p_val = stats.ttest_ind(group_silent, group_noise, equal_var=False)
    else:
        ttest_val, p_val = stats.ttest_ind(group_silent, group_noise, equal_var=True)

    print(f"Levene's test: {t_levene}, {p_levene} | t-test -> {ttest_val}, p = {p_val}")

    # save log as new row in csv file named `ttest_log.csv`
    if savelog:
        log_csv_path = 'ttest_log.csv'

        if not os.path.exists(log_csv_path):
            with open(log_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['noise', 'task', 'freq', 'feature', 't_levene', 'p_levene', 'ttest_val', 'p_val'])

        with open(log_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)

            if os.stat(log_csv_path).st_size == 0:
                writer.writerow(['noise', 'task', 'freq', 'feature', 't_levene', 'p_levene', 'ttest_val', 'p_val'])

            writer.writerow([noise, task, freq, feature, t_levene, p_levene, ttest_val, p_val])

    return ttest_val, p_val
    

    
        