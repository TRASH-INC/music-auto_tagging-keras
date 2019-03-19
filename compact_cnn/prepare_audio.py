# What you need is a mono-channel, 12kHz-sampled audio signals. 

import librosa
import numpy as np


def compute_samples(audio_paths, sr, duration, mono=True):
    """
    compute the input vector of shape (len(audio_paths), 1, 1, sr*duration)
    parameters
    ----------
    audio_paths: list of paths for the audio files.
                Any format supported by audioread will work.
    """
    data_list = [] 
    for path in audio_paths:
        src_loaded, sr = librosa.load(path, sr=sr, duration=duration, mono=mono) # 
        print(src.shape) # (N, )

        if src.shape[0] < sr * duration:
            print('Concat zeros so that the shape becomes (348000, )')
        if src.shape[0] > sr * duration:
            print('If you set the duration as 29 in the loading function, the shape is probably (348001, ).')
            print('In this case, trim it to make it (348000, ).')
            src = src[:sr*duration]

        # It's pretty done, now the src.shape == (348000, )
        # However, Kapre, the audio preprocessing library expect something like (n_channel, length) 
        #  because I wanted the `ndim` of the signal to be in a consistent format.
        # So,...
        src = src[np.newaxis, :] # now it's (1, 348000)
        data_list.append(src)

    return np.stack(data_list, axis=0)
