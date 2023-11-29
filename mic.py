import datetime
import io
import json as json
import os
import pathlib
import queue
import tempfile
import threading

import accelerate
import click
import numpy as np
import speech_recognition as sr
import torch
import yaml
from pydub import AudioSegment
from rich import print
from rich.traceback import install

install()

accelerator = accelerate.Accelerator()


@click.command()
@click.option(
    "--model_name",
    default="base",
    help="Model to use",
    type=click.Choice(
        ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
    ),
)
@click.option(
    "--input_device_idx",
    default=None,
    help="The index of the device to use for input",
    type=int,
)
@click.option(
    "--verbose",
    default=False,
    help="Whether to print verbose output",
    is_flag=True,
    type=bool,
)
@click.option(
    "--energy", default=300, help="Energy level for mic to detect", type=int
)
@click.option(
    "--dynamic_energy",
    default=False,
    is_flag=True,
    help="Flag to enable dynamic energy",
    type=bool,
)
@click.option(
    "--pause", default=0.8, help="Pause time before entry ends", type=float
)
@click.option(
    "--store_transcription_to",
    default="recordings/",
    help="Where to store transcription at",
    type=str,
)
@click.option(
    "--save_file",
    default=False,
    help="Flag to save file",
    is_flag=True,
    type=bool,
)
def main(
    model_name,
    verbose,
    energy,
    pause,
    dynamic_energy,
    save_file,
    store_transcription_to,
    input_device_idx,
):
    if input_device_idx is None:
        print("Available input devices:")
        devices = sr.Microphone.list_microphone_names()
        for idx, device in enumerate(devices):
            print(f"{idx}: {device}")
        input_device_idx = int(
            input("Enter the index of the device you'd like to use: ")
        )

    temp_dir = tempfile.mkdtemp() if save_file else None

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

    output_dict = {}

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    # load model and processor
    processor = WhisperProcessor.from_pretrained(
        f"openai/whisper-{model_name}"
    )
    audio_model = WhisperForConditionalGeneration.from_pretrained(
        f"openai/whisper-{model_name}"
    )
    audio_model.config.forced_decoder_ids = None

    audio_model = accelerator.prepare(audio_model)
    audio_queue = queue.Queue()
    result_queue = queue.Queue()
    threading.Thread(
        target=record_audio,
        args=(
            audio_queue,
            energy,
            pause,
            dynamic_energy,
            save_file,
            temp_dir,
            input_device_idx,
        ),
    ).start()
    threading.Thread(
        target=transcribe_forever,
        args=(
            audio_queue,
            result_queue,
            audio_model,
            processor,
            verbose,
            save_file,
        ),
    ).start()
    start_time_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    while True:
        current_output = result_queue.get()
        current_date_time = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        output_dict[current_date_time] = current_output
        # save to yaml file
        print(
            f"[bold green]{current_date_time}[/bold green]: {current_output}"
        )
        if store_transcription_to is not None:
            pathlib.Path(store_transcription_to).mkdir(
                parents=True, exist_ok=True
            )
            with open(
                pathlib.Path(store_transcription_to)
                / f"output_{start_time_date}.yaml",
                "w",
            ) as f:
                yaml.dump(output_dict, f)


def record_audio(
    audio_queue,
    energy,
    pause,
    dynamic_energy,
    save_file,
    temp_dir,
    input_device_idx,
):
    # load the speech recognizer and set the initial energy threshold and pause threshold
    r = sr.Recognizer()
    r.energy_threshold = energy
    r.pause_threshold = pause
    r.dynamic_energy_threshold = dynamic_energy

    print(f"Current microphone index: {input_device_idx}")
    with sr.Microphone(
        sample_rate=16000, device_index=input_device_idx
    ) as source:
        print(f"Using microphone: {source.__dict__}")
        print("Trancsribing has began :microphone:")

        i = 0
        while True:
            # get and save audio to wav file
            audio = r.listen(source)
            if save_file:
                data = io.BytesIO(audio.get_wav_data())
                audio_clip = AudioSegment.from_file(data)
                filename = os.path.join(temp_dir, f"temp{i}.wav")
                audio_clip.export(filename, format="wav")
                audio_data = filename
            else:
                torch_audio = torch.from_numpy(
                    np.frombuffer(audio.get_raw_data(), np.int16)
                    .flatten()
                    .astype(np.float32)
                    / 32768.0
                )
                audio_data = torch_audio

            audio_queue.put_nowait(audio_data)
            i += 1


def transcribe_forever(
    audio_queue,
    result_queue,
    audio_model,
    processor,
    verbose,
    save_file,
):
    while True:
        audio_data = audio_queue.get()

        input_features = processor(
            audio_data,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features
        input_features = input_features.to(audio_model.device)
        # generate token ids
        predicted_ids = audio_model.generate(input_features)

        result = processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]

        if not verbose:
            result_queue.put_nowait(result)
        else:
            result_queue.put_nowait(result)

        if save_file:
            os.remove(audio_data)


if __name__ == "__main__":
    main()
