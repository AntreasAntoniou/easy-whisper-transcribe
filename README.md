# Easy Whisper Transcribe

## Overview
Easy Whisper Transcription is an advanced voice-to-text conversion tool leveraging OpenAI's Whisper models. Designed for efficiency and ease of use, it offers real-time audio transcription with support for various input devices and customization options.

## Features
- Real-time voice transcription using OpenAI's Whisper models.
- Continuous audio recording with adjustable sensitivity.
- Dynamic energy threshold for enhanced recognition.
- Command-line interface for easy interaction.
- Options to save transcriptions and audio files.

## Available Models

The Easy Whisper Transcription tool supports the following models from OpenAI's Whisper series:

- **tiny**: 39 million parameters (supports both English-only and Multilingual transcription)
- **base**: 74 million parameters (supports both English-only and Multilingual transcription)
- **small**: 244 million parameters (supports both English-only and Multilingual transcription)
- **medium**: 769 million parameters (supports both English-only and Multilingual transcription)
- **large**: 1550 million parameters (supports Multilingual transcription only)
- **large-v2**: 1550 million parameters (supports Multilingual transcription only)
- **large-v3**: 1550 million parameters (supports Multilingual transcription only)

## Software Stack

The Easy Whisper Transcription tool is built upon a robust and efficient software stack:

- **Python**: For its simplicity and vast ecosystem.
- **OpenAI's Whisper Models**: For high-quality, multilingual voice transcription.
- **PyTorch**: For model development and deployment.
- **Speech Recognition**: To enhance voice recognition capabilities.
- **PyDub**: For handling audio data.
- **NumPy**: For efficient data array processing.
- **Rich**: To enhance the CLI experience.
- **Google Fire**: For automatic CLI generation.
- **Accelerate (by Hugging Face)**: For optimizing model performance.
- **YAML**: For storing transcription outputs.

## Installation

### Prerequisites
- Python 3.6 or later.
- `ffmpeg`: Install via `conda` or your system's package manager. For macOS, use `brew install ffmpeg`.
- `PyAudio`: For macOS, first install `portaudio` using `brew install portaudio` and then `pip install pyaudio`. For Linux, install `portaudio19-dev` first.

### Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/easy-whisper-transcription.git
   cd easy-whisper-transcription
2. **Setup a virtual env**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the script from the command line, providing necessary arguments:

```bash
python app.py --model_name base --verbose True --energy 300
```

## Contributing
Contributions to Easy Whisper Transcribe are welcome! Please read our contributing guidelines to get started.

## License
This project is licensed under the MIT License.

## Acknowledgements
This project utilizes OpenAI's Whisper models for voice transcription. Inspiration was taken from the [whisper-mic](https://github.com/mallorbc/whisper_mic) project that was key to the development of this project.