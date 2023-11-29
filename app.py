import datetime
import io
import os
import pathlib
import queue
import tempfile
import threading

import accelerate
import fire
import numpy as np
import speech_recognition as sr
import torch
import yaml
from pydub import AudioSegment
from rich import print
from rich.traceback import install
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Installing rich traceback for better error visibility
install()

# Initialize accelerator for model optimization
accelerator = accelerate.Accelerator()


def main(
    model_name="base",
    energy: int = 300,
    pause: float = 0.2,
    dynamic_energy: bool = False,
    dynamic_energy_adjustment_damping: float = 0.15,
    dynamic_energy_ratio: float = 1.5,
    operation_timeout: float = 1.0,
    phrase_threshold: float = 0.5,
    non_speaking_duration: float = 0.1,
    save_file=False,
    store_transcription_to="recordings/",
    input_device_idx=None,
):
    """
    Main function to handle voice transcription using Whisper models.

    Args:
    model_name (str): The name of the Whisper model to use.
    energy (int): Energy level for mic sensitivity.
    pause (float): Pause duration to consider the end of a phrase.
    dynamic_energy (bool): Flag to enable dynamic energy adjustment.
    save_file (bool): Flag to save the recorded audio files.
    store_transcription_to (str): Path to store transcriptions.
    input_device_idx (int): Index of the input device (microphone).
    """
    # Choose the input device if not specified
    if input_device_idx is None:
        input_device_idx = choose_input_device()

    # Create a temporary directory for saving files if save_file is True
    temp_dir = tempfile.mkdtemp() if save_file else None

    # Check MPS availability for PyTorch if necessary
    check_mps_availability()

    output_dict = {}

    # Load model and processor
    processor, audio_model = load_model(model_name)

    # Setting up audio and result queues
    audio_queue, result_queue = setup_queues()

    # Starting threads for recording and transcription
    start_threads(
        audio_queue=audio_queue,
        result_queue=result_queue,
        audio_model=audio_model,
        processor=processor,
        save_file=save_file,
        temp_dir=temp_dir,
        input_device_idx=input_device_idx,
        energy=energy,
        pause=pause,
        dynamic_energy=dynamic_energy,
        dynamic_energy_adjustment_damping=dynamic_energy_adjustment_damping,
        dynamic_energy_ratio=dynamic_energy_ratio,
        operation_timeout=operation_timeout,
        phrase_threshold=phrase_threshold,
        non_speaking_duration=non_speaking_duration,
    )

    # Process and store transcription results
    process_transcriptions(result_queue, output_dict, store_transcription_to)


# Additional functions will follow in the next parts
def choose_input_device():
    """
    Allows the user to choose an input device if not specified.

    Returns:
    int: Index of the selected input device.
    """
    print("Available input devices:")
    devices = sr.Microphone.list_microphone_names()
    for idx, device in enumerate(devices):
        print(f"{idx}: {device}")
    return int(input("Enter the index of the device you'd like to use: "))


def check_mps_availability():
    """
    Checks and prints the availability of MPS for PyTorch.
    """
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print(
                "MPS not available because the current PyTorch install was not "
                "built with MPS enabled."
            )
        else:
            print(
                "MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine."
            )


def load_model(model_name: str):
    """
    Load the Whisper model and processor based on the model name.

    Args:
    model_name (str): The name of the Whisper model to use.

    Returns:
    Tuple[WhisperProcessor, WhisperForConditionalGeneration]: Loaded processor and model.
    """
    processor = WhisperProcessor.from_pretrained(
        f"openai/whisper-{model_name}"
    )
    audio_model = WhisperForConditionalGeneration.from_pretrained(
        f"openai/whisper-{model_name}"
    )
    audio_model = accelerator.prepare(audio_model)
    return processor, audio_model


def setup_queues():
    """
    Setup audio and result queues for managing data flow.

    Returns:
    Tuple[queue.Queue, queue.Queue]: Audio and result queues.
    """
    audio_queue = queue.Queue()
    result_queue = queue.Queue()
    return audio_queue, result_queue


def start_threads(
    audio_queue: queue.Queue,
    result_queue: queue.Queue,
    audio_model: WhisperForConditionalGeneration.from_pretrained,
    processor: WhisperProcessor.from_pretrained,
    save_file: bool,
    temp_dir: str,
    input_device_idx: int,
    energy: int = 300,
    pause: float = 0.8,
    dynamic_energy: bool = False,
    dynamic_energy_adjustment_damping: float = 0.15,
    dynamic_energy_ratio: float = 1.5,
    operation_timeout: float = 1.0,
    phrase_threshold: float = 0.5,
    non_speaking_duration: float = 0.5,
):
    """
    Start threads for audio recording and transcription.

    Args:
    audio_queue (queue.Queue): Queue for audio data.
    result_queue (queue.Queue): Queue for transcription results.
    audio_model (WhisperForConditionalGeneration): Whisper model.
    processor (WhisperProcessor): Whisper processor.
    save_file (bool): Flag to save the recorded audio files.
    temp_dir (str): Path to temporary directory.
    energy (int): Energy level for mic sensitivity.
    pause (float): Pause duration to consider the end of a phrase.
    dynamic_energy (bool): Flag to enable dynamic energy adjustment.
    input_device_idx (int): Index of the input device (microphone).

    """
    threading.Thread(
        target=record_audio,
        kwargs=dict(
            audio_queue=audio_queue,
            temp_dir=temp_dir,
            input_device_idx=input_device_idx,
            energy=energy,
            pause=pause,
            dynamic_energy=dynamic_energy,
            dynamic_energy_adjustment_damping=dynamic_energy_adjustment_damping,
            dynamic_energy_ratio=dynamic_energy_ratio,
            operation_timeout=operation_timeout,
            phrase_threshold=phrase_threshold,
            non_speaking_duration=non_speaking_duration,
            save_file=save_file,
        ),
        daemon=True,
    ).start()

    threading.Thread(
        target=transcribe_forever,
        args=(
            audio_queue,
            result_queue,
            audio_model,
            processor,
            save_file,
        ),
        daemon=True,
    ).start()


def process_transcriptions(result_queue, output_dict, store_transcription_to):
    """
    Process and store transcription results.

    Args:
    result_queue (queue.Queue): Queue containing transcription results.
    output_dict (dict): Dictionary to store transcription results with timestamps.
    store_transcription_to (str): Path to store transcription results.
    """
    start_time_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    while True:
        current_output = result_queue.get()
        current_date_time = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        output_dict[current_date_time] = current_output

        # Save to yaml file
        if store_transcription_to:
            pathlib.Path(store_transcription_to).mkdir(
                parents=True, exist_ok=True
            )
            with open(
                pathlib.Path(store_transcription_to)
                / f"output_{start_time_date}.yaml",
                "w",
            ) as f:
                yaml.dump(output_dict, f)
            print(
                f"[bold green]{current_date_time}[/bold green]: {current_output}"
            )


def record_audio(
    audio_queue: queue.Queue,
    temp_dir: str,
    input_device_idx: int,
    energy: int = 300,
    pause: float = 0.8,
    dynamic_energy: bool = False,
    dynamic_energy_adjustment_damping: float = 0.15,
    dynamic_energy_ratio: float = 1.5,
    operation_timeout: float = 1.0,
    phrase_threshold: float = 0.5,
    non_speaking_duration: float = 0.5,
    save_file: bool = False,
) -> None:
    """
    Record audio continuously and put it in a queue for transcription.

    Args:
    audio_queue (queue.Queue): Queue to put recorded audio data.
    energy (int): Energy level for mic sensitivity.
    pause (float): Pause duration to consider the end of a phrase.
    dynamic_energy (bool): Flag to enable dynamic energy adjustment.
    save_file (bool): Flag to save the recorded audio files.
    temp_dir (str): Path to temporary directory.
    input_device_idx (int): Index of the input device (microphone).
    """
    r = sr.Recognizer()
    r.energy_threshold = (
        energy  # minimum audio energy to consider for recording
    )
    r.dynamic_energy_threshold = dynamic_energy
    r.dynamic_energy_adjustment_damping = dynamic_energy_adjustment_damping
    r.dynamic_energy_ratio = dynamic_energy_ratio
    r.pause_threshold = pause  # seconds of non-speaking audio before a phrase is considered complete
    r.operation_timeout = operation_timeout  # seconds after an internal operation (e.g., an API request) starts before it times out, or ``None`` for no timeout

    r.phrase_threshold = phrase_threshold  # minimum seconds of speaking audio before we consider the speaking audio a phrase - values below this are ignored (for filtering out clicks and pops)
    r.non_speaking_duration = non_speaking_duration  # seconds of non-speaking audio to allow before a phrase is considered complete

    print(f"Current microphone index: {input_device_idx}")
    with sr.Microphone(
        sample_rate=16000, device_index=input_device_idx
    ) as source:
        print(f"Using microphone: {source.__dict__}")
        print("Transcribing has begun :microphone:")

        i = 0
        while True:
            audio = r.listen(source)
            if save_file:
                data = io.BytesIO(audio.get_wav_data())
                audio_clip = AudioSegment.from_file(data)
                filename = os.path.join(temp_dir, f"temp{i}.wav")
                audio_clip.export(filename, format="wav")
                audio_data = filename
            else:
                audio_data = (
                    np.frombuffer(audio.get_raw_data(), np.int16)
                    .flatten()
                    .astype(np.float32)
                    / 32768.0
                )
                audio_data = torch.from_numpy(audio_data)

            audio_queue.put_nowait(audio_data)
            i += 1


def transcribe_forever(
    audio_queue: queue.Queue,
    result_queue: queue.Queue,
    audio_model: WhisperForConditionalGeneration.from_pretrained,
    processor: WhisperProcessor.from_pretrained,
    save_file: bool,
):
    """
    Continuously transcribe audio data from the queue.
    """
    while True:
        audio_data = audio_queue.get()

        input_features = processor(
            audio_data, sampling_rate=16000, return_tensors="pt"
        ).input_features
        input_features = input_features.to(audio_model.device)

        predicted_ids = audio_model.generate(input_features)
        result = processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]

        result_queue.put_nowait(result)

        if save_file and isinstance(audio_data, str):
            os.remove(audio_data)


if __name__ == "__main__":
    fire.Fire(main)
