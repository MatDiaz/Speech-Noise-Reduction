import torch
import torchaudio
import torchaudio.functional as F
from pathlib import Path

def add_noise(input_folder, output_folder, snr_level):
  print(f"Procesando audios con snr {snr_level} dB")
  input_folder = Path(input_folder)
  output_folder = Path(output_folder + " " + str(snr_level) + " SNR")
  output_folder.mkdir(parents=True, exist_ok=True)

  audio_files = input_folder.rglob("*.flac")

  for each_audio in audio_files:
    data, sr = torchaudio.load(each_audio)
    noise = (torch.rand(data.shape) * 1.3) - 1
    noisy_signal = F.add_noise(data, noise, torch.Tensor([snr_level]))
    full_path = output_folder.joinpath(each_audio.name)
    torchaudio.save(full_path, noisy_signal, sr)
  print(f"finalizando audios con snr {snr_level} dB")