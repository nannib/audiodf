DISCLAIMER
This software is an experiment; it is based on an idea and multiple audio tests in order to explore a possible empirical method to measure the verisimilitude between real and cloned voices.

# audiodf
This program can detect if an audio message is a Deep Fake or it is genuine
## How to Use the Audio Deepfake Detector 

**Requirements:**

- Python >= 3.8
- Librosa library

**Steps:**

1. Install the required libraries:
```bash
pip install -r requirements.txt
```

2. Place at least one sample audio file (*.wav) in the same directory as the `audiodf.py` script. These samples should be recordings of the same phrase spoken by a human.

3. Place the generated voice-cloned (Deepfake) audio file of the same phrase as the sample audio files and name it `test_audio.wav`.

4. Run the program:
```bash
python audiodf.py
```

**Description:**

The program extracts the following features using the Librosa library: chroma_stft, chroma_cqt, chroma_cens, tonnetz, mel, mfcc, rmse, zcr, spectral_centroid, spectral_bandwidth, spectral_contrast, and spectral_rolloff. It also adjusts n_fft and hop_length and performs audio volume normalization.

- Zeros are padded to uniform the lengths of the files.
- The average of the features extracted from the various samples is calculated.
- The cosine distance between the average of the sample features and those extracted from the `test_audio.wav` file is calculated.
- Finally, a comparison is made and a prediction is based on the preset acceptance threshold.

**Output:**

The program will output whether the `test_audio.wav` file is considered genuine or a deepfake based on the calculated distance and the preset threshold.

**Additional Notes:**

- The preset threshold is set to 75% (0.75). If the distance is greater than this threshold, the file is considered genuine. If the distance is lower than this threshold, the file is considered a deepfake.
- You can adjust the threshold value by modifying the `threshold` variable in the `audiodf.py` script.
- The program can be used to detect deepfakes of different voices and speaking styles. However, its accuracy may vary depending on the quality of the deepfake and the similarity of the voice to the sample audio files.

**ITALIAN** 

DISCLAIMER
Questo software è un esperimento; è basato su un’idea e da molteplici test audio al fine di esplorare un metodo empirico possibile per misurare la verosimiglianza fra voci reali e clonate.

Funziona con versioni di Python >= 3.8

1) pip install -r requirements.txt
2) Mettere i file campione (minimo 1) *.wav nella stessa directory del file audiodf.py, i file campione sono le registrazioni di una stessa frase, fatte da un essere umano.
3) Mettere il file generato dal voice cloning (Deep Fake), della stessa frase dei campioni audio e nominarlo test_audio.wav
4) lanciare il programma:
   python audiodf.py

Il programma ha impostata una soglia minima al 75% (0.75), se il file è superiore a quella soglia allora è considerato genuino, se inferiore è considerato Deep Fake.

**DESCRIZIONE**
Il programma estrae le seguenti features tramite la libreria LIBROSA:
chroma_stft, chroma_cqt, chroma_cens, tonnetz, mel, mfcc, rmse, zcr, spectral_centroid, spectral_bandwidth, spectral_contrast, e spectral_rolloff. 
Inoltre, regola n_fft e hop_length e fa una normalizzazione del volume audio.

Fa un padding di zeri per uniformare le lunghezze dei file

Calcola la media delle caratteristiche estratte dai vari campioni
Calcola la distanza coseno tra la media delle caratteristiche dei campioni e quelle estratte dal file test_audio.wav e ne calcola la distanza.
Infine fa il confronto e la previsione basata sulla soglia d'accettazione pre-impostata.
