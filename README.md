# audiodf
This program can detect if an audio message is a Deep Fake or it is genuine

1) pip install -r requirements.txt
2) Mettere i file campione (minimo 1) *.wav nella stessa directory del file audiodf.py, i file campione sono le registrazioni di una stessa frase, fatte da un essere umano.
3) Mettere il file generato dal voice cloning (Deep Fake), della stessa frase dei campioni audio e nominarlo test_audio.wav
4) lanciare il programma:
   python audiodf.py

Il programma ha impostata una soglia minima al 51% (0.51), se il file è superiore a quella soglia allora è considerato genuino, se inferiore è considerato Deep Fake.

DESCRIZIONE
Il programma estrae le seguenti features tramite la libreria LIBROSA:
chroma_stft, chroma_cqt, chroma_cens, tonnetz, mfcc, rmse, zcr, spectral_centroid, spectral_bandwidth, spectral_contrast, e spectral_rolloff. Inoltre, regola n_fft e hop_length

Fa un padding di zeri per uniformare le lunghezze dei file

Calcola la media delle caratteristiche estratte dai vari campioni
Calcola la distanza coseno tra le caratteristiche dei campioni e quelle estratte dal file test_audio.wav e ne calcola la media.
Infine fa il confronto e la previsione basata sulla soglia d'accettazione pre-impostata.
