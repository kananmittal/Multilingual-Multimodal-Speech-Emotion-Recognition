import torchaudio
import torch
import math

def load_audio(path, sr=16000, max_length=30):
    """
    Load audio file with error handling and length normalization
    max_length: maximum length in seconds
    """
    # Fix path: prepend 'datasets/' if not already present
    if not path.startswith('datasets/'):
        path = f"datasets/{path}"
    
    try:
        waveform, orig_sr = torchaudio.load(path)
        
        # Downmix to mono if needed
        if waveform.dim() == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Ensure shape [1, T]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Resample if needed
        if orig_sr != sr:
            waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
        
        # Trim to max_length
        max_samples = sr * max_length
        if waveform.size(1) > max_samples:
            waveform = waveform[:, :max_samples]
        
        # Ensure minimum length (pad if too short)
        min_samples = sr * 0.5  # minimum 0.5 seconds
        if waveform.size(1) < min_samples:
            # Pad with zeros
            pad_length = int(min_samples - waveform.size(1))
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        
        # Return [T] float32
        return waveform.squeeze(0).float()
        
    except Exception as e:
        print(f"Error loading {path}: {e}")
        # Return a dummy audio of 1 second
        return torch.zeros(sr, dtype=torch.float32)


def speed_perturb(waveform: torch.Tensor, factor: float) -> torch.Tensor:
    """
    Change speed by resampling time axis. waveform: [T]
    """
    if abs(factor - 1.0) < 1e-3:
        return waveform
    orig_len = waveform.size(0)
    new_len = int(orig_len / factor)
    waveform = waveform.unsqueeze(0)  # [1, T]
    resampled = torchaudio.functional.resample(waveform, orig_freq=16000, new_freq=int(16000 * factor))
    resampled = torchaudio.functional.resample(resampled, orig_freq=int(16000 * factor), new_freq=16000)
    resampled = resampled.squeeze(0)
    return resampled


def add_noise_snr(waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Add Gaussian noise at target SNR (dB). waveform: [T]
    """
    signal_power = waveform.pow(2).mean().clamp(min=1e-12)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = (signal_power / snr_linear).item()
    noise = torch.randn_like(waveform) * math.sqrt(noise_power)
    return (waveform + noise).clamp(min=-1.0, max=1.0)
