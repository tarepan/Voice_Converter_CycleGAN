import librosa
import numpy as np
import os
import pyworld

def load_wavs(wav_dir, sr):

    wavs = list()
    for file in os.listdir(wav_dir):
        file_path = os.path.join(wav_dir, file)
        wav, _ = librosa.load(file_path, sr = sr, mono = True)
        #wav = wav.astype(np.float64)
        wavs.append(wav)

    return wavs  # [[1,2,5,2,3,5,....(raw_audio)], [wav2], ...] (each raw_audio is np.ndarray)

# single contents:
# np.ndarray(1, T) -> np.ndarray(?, ?) , spectrogram (timelapse of spectral envelopes)
def world_decompose(wav, fs, frame_period = 5.0):

    # Decompose speech signal into f0, spectral envelope and aperiodicity using WORLD
    wav = wav.astype(np.float64) # np.ndarray -> np.ndarray(number is float64)
    f0, timeaxis = pyworld.harvest(wav, fs, frame_period = frame_period, f0_floor = 71.0, f0_ceil = 800.0)
    # np.ndarray(1, T) -> np.ndarray(?, ?) , spectrogram (timelapse of spectral envelopes, spuared magnitude)
    # I suspect np.ndarray(?, ?) is np.ndarray(frequency, frame)
    # https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder/blob/master/pyworld/pyworld.pyx#L299
    # wav: ndarray
    # f0: ndarray
    # timeaxis: ndarray - temporal_positions of each frame
    # fs: int - samling rate [Hz]
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap = pyworld.d4c(wav, f0, timeaxis, fs)

    return f0, timeaxis, sp, ap

def world_encode_spectral_envelop(sp, fs, dim = 24):

    # Get Mel-cepstral coefficients (MCEPs)

    #sp = sp.astype(np.float64)
    coded_sp = pyworld.code_spectral_envelope(sp, fs, dim)  # return - coded_spectral_envelope : ndarray, Coded spectral envelope.

    return coded_sp # ndarray

def world_decode_spectral_envelop(coded_sp, fs):

    fftlen = pyworld.get_cheaptrick_fft_size(fs)
    #coded_sp = coded_sp.astype(np.float32)
    #coded_sp = np.ascontiguousarray(coded_sp)
    decoded_sp = pyworld.decode_spectral_envelope(coded_sp, fs, fftlen)

    return decoded_sp

# transform wavs into acoustic_features
# single contents: np.ndarray(1, T) -> np.ndarray(?, ?) coded_spectral_envelope
def world_encode_data(wavs, fs, frame_period = 5.0, coded_dim = 24):

    f0s = list()
    timeaxes = list()
    sps = list()
    aps = list()
    coded_sps = list()

    for wav in wavs:
        # np.ndarray(1, T) -> np.ndarray(?, ?) I suspect np.ndarray(frequency, frame) , spectrogram (timelapse of spectral envelopes)
        f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = fs, frame_period = frame_period)
        # np.ndarray(?, ?) spectrogram -> np.ndarray(?, ?) coded_spectral_envelope (MCEPs)
        # I suspect, np.ndarray(frequency, frame) spectrogram -> np.ndarray(24(MCEPs), T/frame_period) coded_spectral_envelope (MCEPs)
        coded_sp = world_encode_spectral_envelop(sp = sp, fs = fs, dim = coded_dim)
        f0s.append(f0)
        timeaxes.append(timeaxis)
        sps.append(sp)
        aps.append(ap)
        coded_sps.append(coded_sp)

    return f0s, timeaxes, sps, aps, coded_sps

# lst: Python list : [data1, data2, data3, data4, ...]
def transpose_in_list(lst):

    transposed_lst = list()
    for array in lst:# array == single data (from wav)
        transposed_lst.append(array.T)
                # https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.ndarray.T.html#numpy.ndarray.T
    return transposed_lst


def world_decode_data(coded_sps, fs):

    decoded_sps =  list()

    for coded_sp in coded_sps:
        decoded_sp = world_decode_spectral_envelop(coded_sp, fs)
        decoded_sps.append(decoded_sp)

    return decoded_sps


def world_speech_synthesis(f0, decoded_sp, ap, fs, frame_period):

    #decoded_sp = decoded_sp.astype(np.float64)
    wav = pyworld.synthesize(f0, decoded_sp, ap, fs, frame_period)
    # Librosa could not save wav if not doing so
    wav = wav.astype(np.float32)

    return wav


def world_synthesis_data(f0s, decoded_sps, aps, fs, frame_period):

    wavs = list()

    for f0, decoded_sp, ap in zip(f0s, decoded_sps, aps):
        wav = world_speech_synthesis(f0, decoded_sp, ap, fs, frame_period)
        wavs.append(wav)

    return wavs

# single contents:
# np.ndarray(?, ?) ->
# I suspect from .shape, np.ndarray(24(MCEPs), T/frame_period) ->
def coded_sps_normalization_fit_transoform(coded_sps):
    # calculation trick for mean and std (cancate then calculate)
    # I suspect from .shape,
    # np.ndarray(24(MCEPs), T/frame_period).concat
    #        wav1_t0, t1, t2, ......., wav2_t1, ....
    # MCEP0
    # MCEP1
    # MCEP2
    # MCEP3
    # MCEP4
    # ...
    coded_sps_concatenated = np.concatenate(coded_sps, axis = 1)
    print("coded_sps_concatenated:")
    print(coded_sps_concatenated.shape) # (24, 112398)
    coded_sps_mean = np.mean(coded_sps_concatenated, axis = 1, keepdims = True)
    coded_sps_std = np.std(coded_sps_concatenated, axis = 1, keepdims = True)
    print("coded_sps_mean:")
    print(coded_sps_mean)
    print(coded_sps_mean.shape) # (24, 1)
    print("coded_sps_std:")
    print(coded_sps_std)
    # normalization is applied per dimension (from article)
    coded_sps_normalized = list()
    for coded_sp in coded_sps:
        # normalize each MCEP_timeseries sample with coded_sps_mean
        coded_sps_normalized.append((coded_sp - coded_sps_mean) / coded_sps_std)

    return coded_sps_normalized, coded_sps_mean, coded_sps_std

def coded_sps_normalization_transoform(coded_sps, coded_sps_mean, coded_sps_std):

    coded_sps_normalized = list()
    for coded_sp in coded_sps:
        coded_sps_normalized.append((coded_sp - coded_sps_mean) / coded_sps_std)

    return coded_sps_normalized

def coded_sps_normalization_inverse_transoform(normalized_coded_sps, coded_sps_mean, coded_sps_std):

    coded_sps = list()
    for normalized_coded_sp in normalized_coded_sps:
        coded_sps.append(normalized_coded_sp * coded_sps_std + coded_sps_mean)

    return coded_sps

def coded_sp_padding(coded_sp, multiple = 4):

    num_features = coded_sp.shape[0]
    num_frames = coded_sp.shape[1]
    num_frames_padded = int(np.ceil(num_frames / multiple)) * multiple
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    coded_sp_padded = np.pad(coded_sp, ((0, 0), (num_pad_left, num_pad_right)), 'constant', constant_values = 0)

    return coded_sp_padded

def wav_padding(wav, sr, frame_period, multiple = 4):

    assert wav.ndim == 1
    num_frames = len(wav)
        #                                 (        N_wav_frame / (sr*5 msec)               + 1)
    num_frames_padded = int((np.ceil((np.floor(num_frames / (sr * frame_period / 1000)) + 1) / multiple + 1) * multiple - 1) * (sr * frame_period / 1000))
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    wav_padded = np.pad(wav, (num_pad_left, num_pad_right), 'constant', constant_values = 0)

    return wav_padded


def logf0_statistics(f0s):

    log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
    log_f0s_mean = log_f0s_concatenated.mean()
    log_f0s_std = log_f0s_concatenated.std()

    return log_f0s_mean, log_f0s_std

def pitch_conversion(f0, mean_log_src, std_log_src, mean_log_target, std_log_target):

    # Logarithm Gaussian normalization for Pitch Conversions
    f0_converted = np.exp((np.log(f0) - mean_log_src) / std_log_src * std_log_target + mean_log_target)

    return f0_converted

def wavs_to_specs(wavs, n_fft = 1024, hop_length = None):

    stfts = list()
    for wav in wavs:
        stft = librosa.stft(wav, n_fft = n_fft, hop_length = hop_length)
        stfts.append(stft)

    return stfts


def wavs_to_mfccs(wavs, sr, n_fft = 1024, hop_length = None, n_mels = 128, n_mfcc = 24):

    mfccs = list()
    for wav in wavs:
        mfcc = librosa.feature.mfcc(y = wav, sr = sr, n_fft = n_fft, hop_length = hop_length, n_mels = n_mels, n_mfcc = n_mfcc)
        mfccs.append(mfcc)

    return mfccs


def mfccs_normalization(mfccs):

    mfccs_concatenated = np.concatenate(mfccs, axis = 1)
    mfccs_mean = np.mean(mfccs_concatenated, axis = 1, keepdims = True)
    mfccs_std = np.std(mfccs_concatenated, axis = 1, keepdims = True)

    mfccs_normalized = list()
    for mfcc in mfccs:
        mfccs_normalized.append((mfcc - mfccs_mean) / mfccs_std)

    return mfccs_normalized, mfccs_mean, mfccs_std

# used only in "train.py"
def sample_train_data(dataset_A, dataset_B, n_frames = 128):
    # dataset_X : coded_sps_A_norm (list of normalzied MCEPs from all wav train data)
    # make order-randamized index array (array length == min(len(dataset_A), len(dataset_B)))
    num_samples = min(len(dataset_A), len(dataset_B))
    train_data_A_idx = np.arange(len(dataset_A))
    train_data_B_idx = np.arange(len(dataset_B))
    np.random.shuffle(train_data_A_idx)
    np.random.shuffle(train_data_B_idx)
    train_data_A_idx_subset = train_data_A_idx[:num_samples]
    train_data_B_idx_subset = train_data_B_idx[:num_samples]

    train_data_A = list()
    train_data_B = list()

    for idx_A, idx_B in zip(train_data_A_idx_subset, train_data_B_idx_subset):
        # select data A
        data_A = dataset_A[idx_A]
        # cut out part?
        frames_A_total = data_A.shape[1]
        assert frames_A_total >= n_frames
        start_A = np.random.randint(frames_A_total - n_frames + 1)  # single int between 0 and (frames_A_total - n_frames + 1)
        end_A = start_A + n_frames # == start_A + 128 (can be configured, but seems to be not used)
        train_data_A.append(data_A[:,start_A:end_A])

        data_B = dataset_B[idx_B]
        frames_B_total = data_B.shape[1]
        assert frames_B_total >= n_frames
        start_B = np.random.randint(frames_B_total - n_frames + 1)
        end_B = start_B + n_frames
        train_data_B.append(data_B[:,start_B:end_B])

    train_data_A = np.array(train_data_A)
    train_data_B = np.array(train_data_B)

    return train_data_A, train_data_B
