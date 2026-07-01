import whisper

_model = None

def transcribe_audio(audio_path, model_name="base"):
    global _model

    if _model is None:
        _model = whisper.load_model(model_name)

    result = _model.transcribe(audio_path)
    return result["text"]