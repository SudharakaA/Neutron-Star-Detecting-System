import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, butter, filtfilt
from scipy.fft import fft, fftfreq
from astropy.io import fits

# Function to load and normalize radio signal data
def load_and_normalize_fits(file_path):
    with fits.open(file_path) as file:
        data = file[0].data  # Access the radio signal data
    # Normalize data to zero mean and unit variance
    normalized_data = (data - np.mean(data)) / np.std(data)
    return normalized_data

# Function to apply Fourier Transform and plot frequency spectrum
def plot_frequency_spectrum(data, sampling_rate):
    # Apply Fourier Transform
    frequency_spectrum = fft(data)
    frequencies = fftfreq(len(data), d=1/sampling_rate)
    
    # Plot frequency spectrum
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, np.abs(frequency_spectrum))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Frequency Spectrum of Radio Signal")
    plt.xlim(0, sampling_rate / 2)  # Only show positive frequencies
    plt.show()

# Function to apply autocorrelation and plot result
def plot_autocorrelation(data):
    # Calculate autocorrelation
    autocorrelation = correlate(data, data, mode='full')
    lags = np.arange(-len(data) + 1, len(data))
    
    # Plot autocorrelation function
    plt.figure(figsize=(12, 6))
    plt.plot(lags, autocorrelation)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title("Autocorrelation of Radio Signal")
    plt.show()

# Function to apply bandpass filter
def bandpass_filter(data, lowcut, highcut, sampling_rate, order=5):
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Main function to analyze the radio signal
def analyze_radio_signal(file_path, sampling_rate, lowcut=0.5, highcut=10.0):
    # Load and normalize data
    data = load_and_normalize_fits(file_path)
    
    # Plot original data
    plt.figure(figsize=(12, 6))
    plt.plot(data)
    plt.xlabel("Time")
    plt.ylabel("Signal Amplitude")
    plt.title("Original Radio Signal Data")
    plt.show()
    
    # Plot frequency spectrum
    print("Plotting Frequency Spectrum...")
    plot_frequency_spectrum(data, sampling_rate)
    
    # Apply and plot autocorrelation
    print("Plotting Autocorrelation...")
    plot_autocorrelation(data)
    
    # Apply bandpass filter
    print("Applying Bandpass Filter...")
    filtered_data = bandpass_filter(data, lowcut, highcut, sampling_rate)
    
    # Plot filtered data
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_data)
    plt.xlabel("Time")
    plt.ylabel("Signal Amplitude")
    plt.title("Filtered Radio Signal Data")
    plt.show()
    
    # Re-plot frequency spectrum for filtered data
    print("Plotting Frequency Spectrum for Filtered Data...")
    plot_frequency_spectrum(filtered_data, sampling_rate)
    
    # Apply and plot autocorrelation on filtered data
    print("Plotting Autocorrelation for Filtered Data...")
    plot_autocorrelation(filtered_data)

# Parameters
file_path = 'radio_data.fits'  # Replace with your FITS file path
sampling_rate = 100  # Sampling rate in Hz (adjust according to data)

# Run the analysis
analyze_radio_signal(file_path, sampling_rate)
