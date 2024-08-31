import os
import time

import whisperx

if __name__ == "__main__":
    device = "cuda"
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

    start = time.time()
    model = whisperx.load_model(
        model_name, device, compute_type=compute_type, download_root=model_dir
    )
    print(f"load_model: {time.time() - start:.2f}s")

    start = time.time()
    audio = whisperx.load_audio(audio_file)
    print(f"load_audio: {time.time() - start:.2f}s")

    start = time.time()
    result = model.transcribe(audio, batch_size=batch_size)
    print(f"transcribe_audio: {time.time() - start:.2f}s")
    print(result["segments"])  # before alignment

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
