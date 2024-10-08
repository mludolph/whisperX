import os

import whisperx


def test_transcribe_base() -> None:
    device = "cpu"
    model_name = "large-v3"
    audio_file = "tests/data/test.mov"
    batch_size = 16  # reduce if low on GPU mem
    compute_type = "float32"  # change to "int8" if low on GPU mem (may reduce accuracy)
    hf_access_token = os.environ.get("HF_ACCESS_TOKEN")

    if not hf_access_token:
        raise RuntimeError("huggingface access token missing")

    # 1. Transcribe with original whisper (batched)
    # model = whisperx.load_model(model_name, device, compute_type=compute_type)

    # save model to local path (optional)
    model_dir = "models/"
    model = whisperx.load_model(
        model_name, device, compute_type=compute_type, download_root=model_dir
    )

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    print(result["segments"])  # before alignment

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    print(result["segments"])  # after alignment

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

    # 3. Assign speaker labels
    diarize_model = whisperx.DiarizationPipeline(device=device)

    # add min/max number of speakers if known
    diarize_segments = diarize_model(audio)
    # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

    result = whisperx.assign_word_speakers(diarize_segments, result)
    print(diarize_segments)
    print(result["segments"])  # segments are now assigned speaker IDs
