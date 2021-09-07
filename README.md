# deep.audio
Audio analysis using machine learning

In this repo we'll explore using PyTorch for deep learning audio classification.  We'll use on the fly computation for the time to frequency domain conversion as the model is training. 

# Dataset
We'll use the Urban Sounds 8k dataset for this repo.  Feel free to replace this dataset with your own.  The dataset is available for free here https://urbansounddataset.weebly.com/ and contains 10 audio classes with over 8000 audio samples. Once you have downloaded the compressed dataset, extract it to your current working directory. There is a csv files that contain metadata of all the sound wave metadata.

Alternatively, the dataset is also available on Kaggle https://www.kaggle.com/chrisfilo/urbansound8k/download.

The dataset contains 10 audio classes with the following class labels:

0 = airconditioner 
1 = carhorn
2 = childrenplaying 
3 = dogbark
4 = drilling
5 = engineidling 
6 = gunshot
7 = jackhammer
8 = siren
9 = street_musicThe Urban Sounds 8k 

We'll use uncompressed audio in the WAV format as input. 

# Background

## Fourier Transform
When audio is sampled it is typically at a high rate, for example 44.1 kHz.  This is 44,100 samples per second.  Additionaly the individual samples of audio are represented with 16 bits, so 2^16 = 65536 different integer values.  Audio is sampled in the time domain and this makes it hard to view when viewed as a function of time.  To make this easier we can convert from time domain to the frequency domain using the Fast Fourier Transform (FFT).  The FFT produces a periodogram, where the y-axis is the magnitude and the x-axis is frequency in Hz from 0 to the Nyquest Frequency.  The periodogram is an estimate of the spectral density of the signal

## Nyquest Frequency
This is the highest frequency data you can capture from the audio source.  The Nyquest Frequncy is 0.5 * sampling_rate.  So if you sample data at 44.1 kHz.  The highest frequency you can capture is 22.05 kHz. 

## Short Time Fourier Transform (STFT)
This is a method to create a sliding window across the time series data where each window is used to create a FFT periodogram.  The periodograms are stacked together to create a spectrogram.  To calculate the STFT you need to specify the window length, step size, and number of FFT.  For example if we use data sampled at 16 kHz with a window length of 25 ms ((16000 samples / sec) * (1 sec / 1000 ms) * (25 ms) = 400 samples), and a step size of 10 ms ((16000 samples / sec) * (1 sec / 1000 ms) * (10 ms) = 160 samples).  The window length and step size cause some overlap but a Hamming window is used on the edges of the window.  A decibel spectrogram uses the log base 10 of the spectrogram, the x-axis is time.  The dimension for the x-axis is signal length / step size.  The y-axis is 0.5 of the number of FFT.  The low frequency is at the top with frequency increasing toward the bottom.  

## Mel Filterbank
These are filters that are applied to the frequency ranges.  This basically just re-bins the data.  The low frequency bins are smaller and the high frequency bins are further appart.  The filter bank is applied to the periodogram.  Think of this as rescalling the data from the periodogram to the mel scale using filters.  

## Mel Spectrogram
This is appling the Mel Filterbank to the spectrograms.  Again, just a rebinning.  The number of filter bins effect the ammount of memory used when training.  This can be a hyperparameter.  

