import torch
from scipy.io.wavfile import write
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# Configuración
MODEL_ID = "facebook/musicgen-small"
DEVICE = "cpu"

PROMPT = (
    "Upbeat electronic dance music with energetic synthesizers, "
    "punchy drums, deep bass and a catchy melody, instrumental"
)

# Limitamos los hilos al número de núcleos de tu CPU
torch.set_num_threads(4)

print("Cargando MusicGen...")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)

model = model.to(DEVICE)
model.eval()

print("Preparando prompt...")

inputs = processor(text=[PROMPT], padding=True, return_tensors="pt").to(DEVICE)

print("Generando música...")

with torch.inference_mode():
    audio = model.generate(
        **inputs, do_sample=True, guidance_scale=3.0, max_new_tokens=200
    )

# Guardar audio
sample_rate = model.config.audio_encoder.sampling_rate
waveform = audio[0, 0].cpu().float().numpy()

write("musica.wav", sample_rate, waveform)

print("Música generada: musica.wav")
