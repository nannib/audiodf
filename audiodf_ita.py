# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 18:17:17 2024

@author: Nanni Bassetti - nannibassetti.com
"""
import os
import numpy as np
import pandas as pd
import librosa
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

def get_adaptive_n_fft(y):
    length = len(y)
    n_fft = 2**int(np.floor(np.log2(length)))
    return min(n_fft, length)

def normalize_audio(y):
    max_amplitude = np.max(np.abs(y))
    return y / max_amplitude

def extract_features(audio_path, max_len=1024):
    y, sr = librosa.load(audio_path, sr=None, res_type='kaiser_best')
    y = normalize_audio(y)
    
    n_fft = get_adaptive_n_fft(y)
    hop_length = n_fft // 2
    
    y_harmonic = librosa.effects.harmonic(y)
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=100, n_fft=n_fft, hop_length=hop_length)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    rmse = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
    print('*', end='', flush=True)
    
    def pad_features(feature, max_len):
        return librosa.util.fix_length(feature, size=max_len)
    
    mfccs = pad_features(mfccs, max_len).flatten()
    chroma_stft = pad_features(chroma_stft, max_len).flatten()
    chroma_cqt = pad_features(chroma_cqt, max_len).flatten()
    chroma_cens = pad_features(chroma_cens, max_len).flatten()
    mel = pad_features(mel, max_len).flatten()
    contrast = pad_features(contrast, max_len).flatten()
    tonnetz = pad_features(tonnetz, max_len).flatten()
    zcr = pad_features(zcr, max_len).flatten()
    spectral_centroid = pad_features(spectral_centroid, max_len).flatten()
    spectral_bandwidth = pad_features(spectral_bandwidth, max_len).flatten()
    spectral_rolloff = pad_features(spectral_rolloff, max_len).flatten()
    rmse = pad_features(rmse, max_len).flatten()
    
    features = np.hstack([mfccs, chroma_stft, chroma_cqt, chroma_cens, mel, contrast, tonnetz, zcr, spectral_centroid, spectral_bandwidth, spectral_rolloff, rmse])
    return features

def main():
    default_threshold = 0.75
    threshold = input(f"Inserisci la soglia di genuinità (default {default_threshold}): ")
    if threshold == '':
        threshold = default_threshold
    else:
        threshold = float(threshold)
    
    current_dir = os.getcwd()
    test_audio_path = os.path.join(current_dir, 'test_audio.wav')
    
    wav_files = [f for f in os.listdir(current_dir) if f.endswith('.wav') and f != 'test_audio.wav']
    
    if not wav_files:
        print("Non ci sono file .wav nella directory corrente.")
        return
    
    feature_labels = ['mfccs', 'chroma_stft', 'chroma_cqt', 'chroma_cens', 'mel', 'contrast', 'tonnetz', 'zcr', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'rmse']
    
    real_features = []
    for wav_file in wav_files:
        file_path = os.path.join(current_dir, wav_file)
        features = extract_features(file_path)
        real_features.append(features)
    
    real_features = np.array(real_features)
    
    # Calcola la media delle caratteristiche dei campioni
    mean_real_features = np.mean(real_features, axis=0)
    
    # Estrae le caratteristiche del file di test
    test_features = extract_features(test_audio_path).reshape(1, -1)
    
    # Calcola la similarità coseno tra la media dei campioni e le caratteristiche del file di test
    similarity_scores = cosine_similarity([mean_real_features], test_features)
    average_similarity = similarity_scores[0, 0]
    
    # Determina se il file di test è un fake
    is_fake = average_similarity < threshold
    verosimiglianza = (1 - average_similarity) * 100
    
    # Stampa delle informazioni
    print()
    print(f"Media delle similarità: {average_similarity * 100:.2f}%")
    print(f"Il file 'test_audio.wav' è probabilmente un deep fake? {'Sì perchè < di' if is_fake else 'No, perchè > di'} soglia impostata: {threshold * 100:.2f}% ")
    print(f"Percentuale di verosimiglianza che 'test_audio.wav' sia un deep fake: {verosimiglianza:.2f}% - {100 - threshold * 100:.2f}% (valore soglia)")
    
    generate_report = input("Desideri generare il report XLSX e un file di testo TXT con le informazioni? (Sì/No): ").lower()
    if generate_report == 'si' or generate_report == 'sì':
        export_feature_comparison(real_features, test_features, feature_labels, threshold, average_similarity, is_fake, verosimiglianza)

def export_feature_comparison(real_features, test_features, feature_labels, threshold, average_similarity, is_fake, verosimiglianza):
    mean_real_features = np.mean(real_features, axis=0)
    test_features = test_features.flatten()
    
    num_features = len(mean_real_features)
    chunk_size = num_features // len(feature_labels)
    
    data = {}
    for i, label in enumerate(feature_labels):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        data[f'{label} campioni'] = mean_real_features[start_idx:end_idx]
        data[f'{label} test'] = test_features[start_idx:end_idx]
    
    df = pd.DataFrame(data)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name_xlsx = f'report_features_{timestamp}.xlsx'
    file_name_txt = f'report_info_{timestamp}.txt'
    
    df.to_excel(file_name_xlsx, index=False)
    print(f"Comparazione delle features salvata nel file {file_name_xlsx}")
    
    with open(file_name_txt, 'w') as f:
        f.write(f"Media delle similarità: {average_similarity * 100:.2f}%\n")
        f.write(f"Il file 'test_audio.wav' è probabilmente un deep fake? {'Sì perchè < di' if is_fake else 'No, perchè > di'} soglia impostata: {threshold*100:.2f}% \n")
        f.write(f"Percentuale di verosimiglianza che 'test_audio.wav' sia un deep fake: {verosimiglianza:.2f}% - {100 - threshold*100:.2f}% (valore soglia)\n")
    print(f"Informazioni salvate nel file {file_name_txt}")

if __name__ == "__main__":
    main()







