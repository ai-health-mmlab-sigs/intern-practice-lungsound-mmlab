import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, hilbert, find_peaks
import numpy as np

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', fs=fs, output='sos')
    return sos

def envelope(signal, rate, threshold=0.01):
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    amplitude_envelope /= np.max(amplitude_envelope)
    return amplitude_envelope > threshold

def count_breaths(envelope, rate, distance):
    peaks, _ = find_peaks(envelope, distance=distance)
    return len(peaks)

def calculate_breath_rate(num_breaths, duration_in_seconds):
    return (num_breaths / duration_in_seconds) * 60

def plot_signals(signals, titles, rate):
    plt.figure(figsize=(15, 5))
    for i, (signal, title) in enumerate(zip(signals, titles), 1):
        plt.subplot(1, len(signals), i)
        plt.plot(signal)
        plt.title(title)
        if i == 1:
            plt.ylabel('Amplitude')
        plt.xlabel('Sample')
    plt.tight_layout()
    plt.show()

def process_audio(file_path, threshold, distance):
    rate, audio = wavfile.read(file_path)
    audio = audio.astype(float)

    sos = butter_bandpass(100, 2000, rate, order=3)
    filtered = sosfilt(sos, audio)

    audio_envelope = envelope(filtered, rate, threshold)
    num_breaths = count_breaths(audio_envelope, rate, distance)
    duration_in_seconds = len(audio) / rate
    breath_rate = calculate_breath_rate(num_breaths, duration_in_seconds)

    plot_signals([filtered, audio_envelope], ['Filtered Signal', 'Envelope'], rate)

    return num_breaths, breath_rate

if __name__ == "__main__":
    file_path = "/home/rlg/projects/sig/mmlab-sigs_practice_lungsound/data/icbhi_dataset/audio_test_data/102_1b1_Ar_sc_Meditron.wav"
    threshold = 0.001 # Adjust this threshold for envelope detection
    distance = 100 # Adjust this distance for peak detection (samples)
    num_breaths, breath_rate = process_audio(file_path, threshold, distance)
    print(f"Number of breaths: {num_breaths}")
    print(f"Breath rate per minute: {breath_rate:.2f}")