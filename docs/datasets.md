Brazilian Portuguese Merged Speech Dataset (Derived from Common Voice)
This dataset is a preprocessed and merged version of the Mozilla Common Voice dataset for Brazilian Portuguese (pt-BR). It was created by filtering, merging, and normalizing audio clips to improve usability for speech recognition and TTS (Text-to-Speech) training.

📌 Dataset Details
Source: Derived from Common Voice Corpus 20.0
Language: 🇧🇷 Brazilian Portuguese (pt-BR)
Format: MP3 (24 kHz, mono, 64 kbps)
Metadata: metadata.csv file with corresponding transcriptions
License: CC BY-SA 3.0
🎯 Preprocessing Steps
This dataset was generated using a Python script that:

Filtered Brazilian Portuguese sentences (removed pt-PT).
Kept only validated recordings (at least one upvote).
Merged short clips from the same speaker (5 to 27 sec per file).
Added 300ms silence between merged sentences.
Normalized the audio:
24 kHz sample rate
16-bit sample width
Mono audio
64 kbps bitrate
Generated a transcription file (metadata.csv).
📂 File Structure
├── wavs/       # Processed audio files (MP3)
│   ├── segment_1.mp3
│   ├── segment_2.mp3
│   ├── ...
├── metadata.csv         # Transcriptions (filename|text)
├── segment_index.json   # Processing progress tracking
├── README.md            # This file

prepare complete samples : 35072 time data : 187:12:22 min sec : 5.06 max sec : 30.0

📜 License & Attribution
This dataset is released under the Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0) license, the same as the original Common Voice dataset.

To comply with this license:

You must give credit to Mozilla Common Voice.
If you modify or redistribute this dataset, you must release it under the same CC BY-SA 3.0 license.
Do not imply endorsement from Mozilla or Common Voice.
📌 Citation Example:
Derived from Mozilla Common Voice, licensed under CC BY-SA 3.0.
https://commonvoice.mozilla.org/

🚀 Usage Examples
Python (Load Metadata)
import pandas as pd

metadata = pd.read_csv("metadata.csv", sep="|")
print(metadata.head())

This is made to be used on F5-TTS, follow instructions from https://github.com/SWivid/F5-TTS/discussions/57 to prepare the dataset.

Using with Speech Models
You can use this dataset for:

Training Speech-to-Text (STT) models (e.g., Whisper, DeepSpeech)
Building Text-to-Speech (TTS) systems
Speaker recognition & voice cloning
💡 Contributing & Feedback
