import os
import time

from faster_whisper import BatchedInferencePipeline, WhisperModel
from faster_whisper.transcribe import TranscriptionOptions

import whisperx

default_asr_options = {
    "beam_size": 5,
    "best_of": 5,
    "patience": 1,
    "length_penalty": 1,
    "repetition_penalty": 1,
    "no_repeat_ngram_size": 0,
    "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "condition_on_previous_text": False,
    "prompt_reset_on_temperature": 0.5,
    "initial_prompt": None,
    "prefix": None,
    "suppress_blank": True,
    "suppress_tokens": [-1],
    "without_timestamps": True,
    "max_initial_timestamp": 0.0,
    "word_timestamps": False,
    "prepend_punctuations": "\"'“¿([{-",
    "append_punctuations": "\"'.。,，!！?？:：”)]}、",
    "max_new_tokens": None,
    "clip_timestamps": None,
    "hallucination_silence_threshold": None,
    "hotwords": None,
    "log_prob_low_threshold": None,
    "multilingual": False,
    "output_language": False,
}
suppress_numerals = False
default_asr_options = TranscriptionOptions(**default_asr_options)
default_vad_options = {"vad_onset": 0.500, "vad_offset": 0.363}


if __name__ == "__main__":
    device = "cuda"
    model_name = "large-v3"
    audio_file = "tests/data/test.mov"
    batch_size = 16  # reduce if low on GPU mem
    compute_type = "float32"  # change to "int8" if low on GPU mem (may reduce accuracy)
    model_dir = "models/"

    # hf_access_token = os.environ.get("HF_ACCESS_TOKEN")
    # if not hf_access_token:
    #     raise RuntimeError("huggingface access token missing")

    # 1. Transcribe with original whisper (batched)
    # model = whisperx.load_model(model_name, device, compute_type=compute_type)

    # save model to local path (optional)
    start = time.time()

    model = WhisperModel(
        model_name, device, compute_type=compute_type, download_root=model_dir
    )
    pipeline = BatchedInferencePipeline(model=model, options=default_asr_options)

    # model = whisperx.load_model(
    # model_name, device, compute_type=compute_type, download_root=model_dir
    # )
    print(f"load_model: {time.time() - start:.2f}s")

    start = time.time()
    audio = whisperx.load_audio(audio_file)
    print(f"load_audio: {time.time() - start:.2f}s")

    start = time.time()
    segments, info = pipeline.transcribe(audio, batch_size=batch_size)

    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

    print(f"transcribe_audio: {time.time() - start:.2f}s")
    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model

    # # 2. Align whisper output
    # start = time.time()
    # model_a, metadata = whisperx.load_align_model(
    #     language_code=result["language"], device=device
    # )
    # print(f"load_align: {time.time() - start:.2f}s")

    # start = time.time()
    # result = whisperx.align(
    #     result["segments"],
    #     model_a,
    #     metadata,
    #     audio,
    #     device,
    #     return_char_alignments=False,
    # )
    # print(f"align_audio: {time.time() - start:.2f}s")

    # print(result["segments"])  # after alignment

    # # delete model if low on GPU resources
    # # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

    # # 3. Assign speaker labels

    # start = time.time()
    # diarize_model = whisperx.DiarizationPipeline(
    #     device=device, use_auth_token=hf_access_token
    # )
    # print(f"load_diarize: {time.time() - start:.2f}s")

    # # add min/max number of speakers if known
    # start = time.time()
    # diarize_segments = diarize_model(audio, num_speakers=2)
    # print(f"diarize_audio: {time.time() - start:.2f}s")

    # # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

    # start = time.time()
    # result = whisperx.assign_word_speakers(diarize_segments, result)
    # print(f"assign_speakers: {time.time() - start:.2f}s")

    # print(diarize_segments)
    # print(result["segments"])  # segments are now assigned speaker IDs
