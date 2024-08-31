import os
import timeit
from math import sqrt

import numpy as np
from scipy import stats

import whisperx
from whisperx.audio import SAMPLE_RATE

if __name__ == "__main__":
    device = "cuda"
    model_name = "distil-large-v3"
    audio_file = "tests/data/test.mov"
    batch_size = 16  # reduce if low on GPU mem
    compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)
    hf_access_token = os.environ.get("HF_ACCESS_TOKEN")

    if not hf_access_token:
        raise RuntimeError("huggingface access token missing")

    # 1. Transcribe with original whisper (batched)
    # model = whisperx.load_model(model_name, device, compute_type=compute_type)

    # save model to local path (optional)
    model_dir = "models/"

    audio = whisperx.load_audio(audio_file)
    model = whisperx.load_model(
        model_name, device, compute_type=compute_type, download_root=model_dir
    )
    language = model.detect_language(audio)

    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    diarize_model = whisperx.DiarizationPipeline(
        device=device, use_auth_token=hf_access_token
    )

    def benchmark() -> None:
        result = model.transcribe(audio, batch_size=batch_size, language=language)
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
        # print(result["segments"])  # after alignment

        diarize_segments = diarize_model(audio, num_speakers=2)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # print(diarize_segments)
        # print(result["segments"])  # segments are now assigned speaker IDs

    print("running benchmark...")
    num_iters = 5
    res = timeit.repeat(benchmark, repeat=num_iters, number=1)
    res_stats = stats.describe(res)
    audio_len = audio.size / SAMPLE_RATE
    real_time_fact = audio_len / res_stats.mean
    print(
        f"transcribe(repeat={num_iters}) {audio_len}s (model_name={model_name}, compute_type={compute_type}, batch_size={batch_size}): {res_stats.mean:.2f}s ± {sqrt(res_stats.variance):.2f}s ({res_stats.minmax[0]:.2f}s - {res_stats.minmax[1]:.2f}s), {real_time_fact:.2f}x"
    )

    # transcribe(repeat=5) 49.4356875s (model_name=large-v3, compute_type=float16, batch_size=16): 4.07s ± 0.30s (3.87s - 4.59s), 12.13905861891849x
