import torch
import torchaudio as ta

from chatterbox.mtl_tts import ChatterboxMultilingualTTS


device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")
print("Loading Chatterbox Multilingual...")

model = ChatterboxMultilingualTTS.from_pretrained(
    device=device
)

print("Model loaded.")

text = "Okay, opening YouTube."

print("Generating audio...")

wav = model.generate(
    text,
    language_id="en"
)

ta.save(
    "chatterbox_test.wav",
    wav,
    model.sr
)

print("Saved: chatterbox_test.wav")