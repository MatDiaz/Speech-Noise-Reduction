import torch
import torchaudio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def denoise_audio(model, noisy_waveform, model_params, noisy_mean, noisy_std, clean_mean, clean_std):
    model.eval()
    noisy_waveform = noisy_waveform.to(device)
    window = torch.hamming_window(model_params["window_length"], periodic=True)

    noisy_stft = torch.stft(noisy_waveform.squeeze(), n_fft=model_params["fft_length"],
                            hop_length=model_params["window_length"] - model_params["overlap"],
                            window=window, return_complex=True)
    noisy_mag = noisy_stft.abs()
    noisy_phase = noisy_stft.angle()  # fase original

    padded = torch.cat([noisy_mag[:, :model_params["num_segments"]-1], noisy_mag], dim=1)
    segments = []
    for i in range(noisy_mag.shape[1]):
        seg = padded[:, i:i+model_params["num_segments"]]
        seg = (seg - noisy_mean) / noisy_std 
        segments.append(seg.unsqueeze(0))
    X_proc = torch.stack(segments, dim=0).unsqueeze(1).to(device)

    with torch.no_grad():
        preds = model(X_proc)
    preds = preds.cpu() * clean_std + clean_mean 

    clean_mag_est = preds.T  # [freq, frames]

    # Combinar con fase original
    clean_stft_est = torch.polar(clean_mag_est, noisy_phase)

    # ISTFT
    clean_waveform = torch.istft(clean_stft_est, n_fft=model_params["fft_length"],
                                 hop_length=model_params["window_length"] - model_params["overlap"],
                                 window=window.cpu())

    return clean_waveform

