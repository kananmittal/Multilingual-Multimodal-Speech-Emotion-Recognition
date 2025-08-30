import torch
import torch.nn as nn
import numpy as np
import librosa
from typing import Dict, Tuple, Optional, List
import logging
from dataclasses import dataclass
import warnings
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch, wiener
import soundfile as sf

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    warnings.warn("noisereduce not available. Using scipy Wiener filtering.")

try:
    import pyloudnorm as pyln
    PYLN_AVAILABLE = True
except ImportError:
    PYLN_AVAILABLE = False
    warnings.warn("pyloudnorm not available. Using basic RMS normalization.")

@dataclass
class ConditioningFeatures:
    """Container for audio conditioning features and metadata."""
    # Processing flags
    hum_filtered: bool
    hpf_applied: bool
    denoise_applied: bool
    dereverb_applied: bool
    
    # Quality metrics
    snr_before: float
    snr_after: float
    denoise_gain_db: float
    estimated_t60: float
    
    # Normalization metrics
    lufs_original: float
    lufs_target: float
    lufs_adjustment: float
    peak_reduction_db: float
    compression_ratio: float
    
    # Filter parameters
    hpf_cutoff: float
    hum_frequencies: List[float]
    noise_type_detected: str
    
    # Additional features for downstream fusion
    conditioning_features: torch.Tensor  # [12] tensor for fusion


class HumNotchFilter:
    """Hum notch filtering for 50Hz and 60Hz power line interference."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.hum_frequencies = [50, 60]  # Hz
        self.q_factor = 30  # Quality factor for notch filters
    
    def detect_hum(self, audio: np.ndarray) -> List[float]:
        """Detect hum frequencies in audio."""
        # Compute power spectral density
        freqs, psd = signal.welch(audio, fs=self.sample_rate, nperseg=2048)
        
        detected_hum = []
        for hum_freq in self.hum_frequencies:
            # Find peak near hum frequency
            freq_idx = np.argmin(np.abs(freqs - hum_freq))
            peak_power = psd[freq_idx]
            
            # Check if peak is significant (above threshold)
            threshold = np.mean(psd) + 2 * np.std(psd)
            if peak_power > threshold:
                detected_hum.append(hum_freq)
        
        return detected_hum
    
    def apply_notch_filters(self, audio: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """Apply notch filters to remove hum frequencies."""
        detected_hum = self.detect_hum(audio)
        filtered_audio = audio.copy()
        
        for hum_freq in detected_hum:
            # Design notch filter
            b, a = iirnotch(hum_freq, self.q_factor, self.sample_rate)
            filtered_audio = filtfilt(b, a, filtered_audio)
        
        return filtered_audio, detected_hum


class HighPassFilter:
    """High-pass filtering for low-frequency noise removal."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.default_cutoff = 80  # Hz
        self.order = 4
    
    def should_apply_hpf(self, audio: np.ndarray) -> Tuple[bool, float]:
        """Determine if high-pass filtering should be applied."""
        # Compute frequency spectrum
        freqs, psd = signal.welch(audio, fs=self.sample_rate, nperseg=2048)
        
        # Calculate low-frequency energy (below 200 Hz)
        low_freq_mask = freqs < 200
        low_freq_energy = np.sum(psd[low_freq_mask])
        total_energy = np.sum(psd)
        
        low_freq_ratio = low_freq_energy / total_energy if total_energy > 0 else 0
        
        # Apply HPF if low-frequency energy > 20%
        should_apply = low_freq_ratio > 0.2
        
        # Determine optimal cutoff frequency
        if should_apply:
            # Find frequency where energy drops significantly
            cumulative_energy = np.cumsum(psd)
            energy_threshold = 0.1 * cumulative_energy[-1]
            cutoff_idx = np.where(cumulative_energy > energy_threshold)[0]
            if len(cutoff_idx) > 0:
                cutoff_freq = freqs[cutoff_idx[0]]
                cutoff_freq = max(80, min(100, cutoff_freq))  # Clamp to 80-100 Hz
            else:
                cutoff_freq = self.default_cutoff
        else:
            cutoff_freq = self.default_cutoff
        
        return should_apply, cutoff_freq
    
    def apply_hpf(self, audio: np.ndarray, cutoff_freq: float) -> np.ndarray:
        """Apply high-pass filter."""
        # Design Butterworth high-pass filter
        nyquist = self.sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        b, a = butter(self.order, normalized_cutoff, btype='high')
        
        # Apply filter with zero-phase filtering
        filtered_audio = filtfilt(b, a, audio)
        
        return filtered_audio


class AdaptiveDenoiser:
    """Adaptive denoising using multiple methods."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.snr_threshold = 15  # dB
    
    def estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate signal-to-noise ratio."""
        # Simple SNR estimation using energy ratio
        energy = np.mean(audio ** 2)
        noise_floor = np.percentile(audio ** 2, 10)
        
        if noise_floor > 0:
            snr_db = 10 * np.log10(energy / noise_floor)
        else:
            snr_db = 50.0  # Very high SNR if no noise detected
        
        return max(0.0, min(50.0, snr_db))
    
    def detect_noise_type(self, audio: np.ndarray) -> str:
        """Detect type of noise present."""
        # Analyze spectral characteristics
        freqs, psd = signal.welch(audio, fs=self.sample_rate, nperseg=1024)
        
        # Check for different noise types
        low_freq_energy = np.sum(psd[freqs < 500])
        mid_freq_energy = np.sum(psd[(freqs >= 500) & (freqs < 2000)])
        high_freq_energy = np.sum(psd[freqs >= 2000])
        
        total_energy = low_freq_energy + mid_freq_energy + high_freq_energy
        
        if total_energy > 0:
            low_ratio = low_freq_energy / total_energy
            mid_ratio = mid_freq_energy / total_energy
            high_ratio = high_freq_energy / total_energy
            
            if low_ratio > 0.5:
                return "low_frequency"
            elif high_ratio > 0.4:
                return "high_frequency"
            elif mid_ratio > 0.6:
                return "mid_frequency"
            else:
                return "white_noise"
        else:
            return "unknown"
    
    def wiener_denoise(self, audio: np.ndarray) -> Tuple[np.ndarray, float]:
        """Apply Wiener filtering for denoising."""
        # Estimate noise from first and last 10% of audio
        noise_samples = int(0.1 * len(audio))
        noise_estimate = np.concatenate([audio[:noise_samples], audio[-noise_samples:]])
        
        # Apply Wiener filter
        denoised_audio = wiener(audio, mysize=len(noise_estimate))
        
        # Calculate gain
        original_energy = np.mean(audio ** 2)
        denoised_energy = np.mean(denoised_audio ** 2)
        
        if denoised_energy > 0:
            gain_db = 10 * np.log10(denoised_energy / original_energy)
        else:
            gain_db = 0.0
        
        return denoised_audio, gain_db
    
    def spectral_gating_denoise(self, audio: np.ndarray) -> Tuple[np.ndarray, float]:
        """Apply spectral gating denoising using noisereduce."""
        if not NOISEREDUCE_AVAILABLE:
            return self.wiener_denoise(audio)
        
        # Estimate noise from non-speech regions
        noise_samples = int(0.1 * len(audio))
        noise_estimate = np.concatenate([audio[:noise_samples], audio[-noise_samples:]])
        
        # Apply spectral gating
        denoised_audio = nr.reduce_noise(
            y=audio, 
            sr=self.sample_rate,
            y_noise=noise_estimate,
            stationary=False
        )
        
        # Calculate gain
        original_energy = np.mean(audio ** 2)
        denoised_energy = np.mean(denoised_audio ** 2)
        
        if denoised_energy > 0:
            gain_db = 10 * np.log10(denoised_energy / original_energy)
        else:
            gain_db = 0.0
        
        return denoised_audio, gain_db
    
    def denoise(self, audio: np.ndarray) -> Tuple[np.ndarray, float, str]:
        """Apply adaptive denoising."""
        snr = self.estimate_snr(audio)
        noise_type = self.detect_noise_type(audio)
        
        if snr < self.snr_threshold:
            # Apply denoising
            if NOISEREDUCE_AVAILABLE:
                denoised_audio, gain_db = self.spectral_gating_denoise(audio)
            else:
                denoised_audio, gain_db = self.wiener_denoise(audio)
        else:
            # No denoising needed
            denoised_audio = audio.copy()
            gain_db = 0.0
        
        return denoised_audio, gain_db, noise_type


class Dereverberator:
    """Dereverberation using Weighted Prediction Error (WPE) method."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.t60_threshold = 0.5  # seconds
        self.segment_length = 4 * sample_rate  # 4 seconds
        self.overlap = 0.5  # 50% overlap
    
    def estimate_t60(self, audio: np.ndarray) -> float:
        """Estimate reverberation time T60."""
        # Simple T60 estimation using energy decay
        # This is a simplified version - in practice, use more sophisticated methods
        
        # Find the peak of the signal
        peak_idx = np.argmax(np.abs(audio))
        
        # Analyze decay after peak
        decay_region = audio[peak_idx:]
        if len(decay_region) < self.sample_rate:
            return 0.1  # Short audio, assume low reverb
        
        # Calculate energy decay
        energy = np.cumsum(decay_region ** 2)
        if energy[-1] == 0:
            return 0.1
        
        # Find time to decay by 60dB
        threshold = energy[-1] * 0.001  # -60dB
        decay_idx = np.where(energy < threshold)[0]
        
        if len(decay_idx) > 0:
            t60 = decay_idx[0] / self.sample_rate
        else:
            t60 = 0.1
        
        return min(t60, 2.0)  # Cap at 2 seconds
    
    def simple_dereverb(self, audio: np.ndarray) -> Tuple[np.ndarray, float]:
        """Simple dereverberation using spectral subtraction."""
        # This is a simplified implementation
        # In practice, use proper WPE or other advanced methods
        
        # Estimate late reverberation
        freqs, psd = signal.welch(audio, fs=self.sample_rate, nperseg=1024)
        
        # Simple spectral subtraction for late reverb
        reverb_estimate = np.mean(psd) * 0.1  # Assume 10% reverb
        psd_clean = np.maximum(psd - reverb_estimate, psd * 0.1)
        
        # Reconstruct signal (simplified)
        # In practice, use proper inverse FFT
        gain = np.sqrt(psd_clean / (psd + 1e-10))
        gain = np.clip(gain, 0.1, 1.0)
        
        # Apply gain (simplified - in practice use proper filtering)
        dereverbed_audio = audio * np.mean(gain)
        
        # Calculate gain
        original_energy = np.mean(audio ** 2)
        dereverbed_energy = np.mean(dereverbed_audio ** 2)
        
        if dereverbed_energy > 0:
            gain_db = 10 * np.log10(dereverbed_energy / original_energy)
        else:
            gain_db = 0.0
        
        return dereverbed_audio, gain_db
    
    def dereverb(self, audio: np.ndarray) -> Tuple[np.ndarray, float]:
        """Apply dereverberation if needed."""
        t60 = self.estimate_t60(audio)
        
        if t60 > self.t60_threshold:
            # Apply dereverberation
            dereverbed_audio, gain_db = self.simple_dereverb(audio)
        else:
            # No dereverberation needed
            dereverbed_audio = audio.copy()
            gain_db = 0.0
        
        return dereverbed_audio, gain_db


class LoudnessNormalizer:
    """Loudness normalization using LUFS."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.target_lufs = -23.0  # Target LUFS level
        self.max_compression_ratio = 4.0
        self.max_gain_change = 6.0  # dB per 100ms
    
    def measure_lufs(self, audio: np.ndarray) -> float:
        """Measure LUFS of audio."""
        if PYLN_AVAILABLE:
            # Use pyloudnorm for accurate LUFS measurement
            meter = pyln.Meter(self.sample_rate)
            loudness = meter.integrated_loudness(audio)
            return loudness
        else:
            # Fallback to RMS-based measurement
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 0:
                lufs = 20 * np.log10(rms) - 70  # Approximate conversion
            else:
                lufs = -60
            return lufs
    
    def apply_compression(self, audio: np.ndarray) -> Tuple[np.ndarray, float]:
        """Apply gentle compression to preserve emotional prosody."""
        # Calculate dynamic range
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.max(np.abs(audio))
        
        if rms > 0:
            dynamic_range_db = 20 * np.log10(peak / rms)
        else:
            dynamic_range_db = 0
        
        # Apply compression if dynamic range > 40dB
        if dynamic_range_db > 40:
            # Simple compression (in practice, use proper compressor)
            threshold = rms * 2  # Compress above 2x RMS
            ratio = min(self.max_compression_ratio, dynamic_range_db / 40)
            
            compressed_audio = audio.copy()
            mask = np.abs(audio) > threshold
            compressed_audio[mask] = np.sign(audio[mask]) * (
                threshold + (np.abs(audio[mask]) - threshold) / ratio
            )
            
            compression_ratio = ratio
        else:
            compressed_audio = audio.copy()
            compression_ratio = 1.0
        
        return compressed_audio, compression_ratio
    
    def normalize_loudness(self, audio: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        """Normalize audio to target LUFS level."""
        # Measure original LUFS
        original_lufs = self.measure_lufs(audio)
        
        # Apply compression if needed
        compressed_audio, compression_ratio = self.apply_compression(audio)
        
        # Calculate required gain
        lufs_adjustment = self.target_lufs - original_lufs
        
        # Limit gain changes to preserve emotional prosody
        max_adjustment = self.max_gain_change
        lufs_adjustment = np.clip(lufs_adjustment, -max_adjustment, max_adjustment)
        
        # Apply gain
        gain_linear = 10 ** (lufs_adjustment / 20)
        normalized_audio = compressed_audio * gain_linear
        
        # Calculate peak reduction
        original_peak = np.max(np.abs(audio))
        normalized_peak = np.max(np.abs(normalized_audio))
        
        if original_peak > 0:
            peak_reduction_db = 20 * np.log10(normalized_peak / original_peak)
        else:
            peak_reduction_db = 0.0
        
        return normalized_audio, lufs_adjustment, peak_reduction_db, compression_ratio


class AudioConditioningModule(nn.Module):
    """
    Audio Conditioning Module.
    
    Intelligent audio preprocessing with policy-driven filtering,
    adaptive denoising, dereverberation, and loudness normalization.
    """
    
    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        
        self.sample_rate = sample_rate
        
        # Initialize conditioning components
        self.hum_filter = HumNotchFilter(sample_rate)
        self.hpf = HighPassFilter(sample_rate)
        self.denoiser = AdaptiveDenoiser(sample_rate)
        self.dereverberator = Dereverberator(sample_rate)
        self.normalizer = LoudnessNormalizer(sample_rate)
        
        # Conditioning feature projection for downstream fusion
        self.conditioning_projection = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 12)
        )
        
        logging.info(f"Audio Conditioning Module initialized for {sample_rate}Hz")
    
    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, ConditioningFeatures]:
        """
        Apply audio conditioning.
        
        Args:
            audio: Audio tensor [batch_size, samples] or [samples]
            
        Returns:
            conditioned_audio: Conditioned audio tensor
            features: Conditioning features and metadata
        """
        # Convert to numpy for processing
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy()
        else:
            audio_np = audio
        
        # Handle batch dimension
        if audio_np.ndim == 2:
            # Process each sample in batch
            conditioned_audios = []
            all_features = []
            
            for i in range(audio_np.shape[0]):
                audio_sample = audio_np[i]
                conditioned_audio, features = self._process_single_sample(audio_sample)
                
                conditioned_audios.append(conditioned_audio)
                all_features.append(features)
            
            # Stack results
            conditioned_audio = torch.stack(conditioned_audios) if conditioned_audios else audio
            features = all_features[0] if all_features else self._create_empty_features()
            
            return conditioned_audio, features
        else:
            # Single sample
            return self._process_single_sample(audio_np)
    
    def _process_single_sample(self, audio: np.ndarray) -> Tuple[torch.Tensor, ConditioningFeatures]:
        """Process a single audio sample through conditioning pipeline."""
        
        # Store original audio for comparison
        original_audio = audio.copy()
        
        # 1. Hum notch filtering
        hum_filtered_audio, hum_frequencies = self.hum_filter.apply_notch_filters(audio)
        hum_filtered = len(hum_frequencies) > 0
        
        # 2. High-pass filtering
        should_apply_hpf, hpf_cutoff = self.hpf.should_apply_hpf(hum_filtered_audio)
        if should_apply_hpf:
            hpf_audio = self.hpf.apply_hpf(hum_filtered_audio, hpf_cutoff)
            hpf_applied = True
        else:
            hpf_audio = hum_filtered_audio.copy()
            hpf_applied = False
            hpf_cutoff = 0.0
        
        # 3. Adaptive denoising
        snr_before = self.denoiser.estimate_snr(hpf_audio)
        denoised_audio, denoise_gain_db, noise_type = self.denoiser.denoise(hpf_audio)
        denoise_applied = denoise_gain_db != 0.0
        snr_after = self.denoiser.estimate_snr(denoised_audio)
        
        # 4. Dereverberation
        estimated_t60 = self.dereverberator.estimate_t60(denoised_audio)
        dereverbed_audio, dereverb_gain_db = self.dereverberator.dereverb(denoised_audio)
        dereverb_applied = dereverb_gain_db != 0.0
        
        # 5. Loudness normalization
        lufs_original = self.normalizer.measure_lufs(dereverbed_audio)
        normalized_audio, lufs_adjustment, peak_reduction_db, compression_ratio = self.normalizer.normalize_loudness(dereverbed_audio)
        lufs_target = self.normalizer.target_lufs
        
        # 6. Create conditioning features
        features = ConditioningFeatures(
            hum_filtered=hum_filtered,
            hpf_applied=hpf_applied,
            denoise_applied=denoise_applied,
            dereverb_applied=dereverb_applied,
            snr_before=snr_before,
            snr_after=snr_after,
            denoise_gain_db=denoise_gain_db,
            estimated_t60=estimated_t60,
            lufs_original=lufs_original,
            lufs_target=lufs_target,
            lufs_adjustment=lufs_adjustment,
            peak_reduction_db=peak_reduction_db,
            compression_ratio=compression_ratio,
            hpf_cutoff=hpf_cutoff,
            hum_frequencies=hum_frequencies,
            noise_type_detected=noise_type,
            conditioning_features=torch.zeros(12)  # Will be set below
        )
        
        # 7. Create conditioning features for downstream fusion
        conditioning_features = torch.tensor([
            float(hum_filtered),
            float(hpf_applied),
            float(denoise_applied),
            float(dereverb_applied),
            snr_before / 50.0,  # Normalize to [0, 1]
            snr_after / 50.0,
            denoise_gain_db / 20.0,  # Normalize to [0, 1]
            estimated_t60 / 2.0,  # Normalize to [0, 1]
            (lufs_original + 60) / 60,  # Normalize to [0, 1]
            lufs_adjustment / 20.0,  # Normalize to [0, 1]
            peak_reduction_db / 20.0,  # Normalize to [0, 1]
            compression_ratio / 4.0  # Normalize to [0, 1]
        ], dtype=torch.float32)
        
        # Project conditioning features
        conditioning_features = self.conditioning_projection(conditioning_features.unsqueeze(0)).squeeze(0)
        features.conditioning_features = conditioning_features
        
        # 8. Return conditioned audio
        conditioned_audio = torch.tensor(normalized_audio, dtype=torch.float32)
        
        return conditioned_audio, features
    
    def _create_empty_features(self) -> ConditioningFeatures:
        """Create empty conditioning features for error cases."""
        return ConditioningFeatures(
            hum_filtered=False,
            hpf_applied=False,
            denoise_applied=False,
            dereverb_applied=False,
            snr_before=0.0,
            snr_after=0.0,
            denoise_gain_db=0.0,
            estimated_t60=0.0,
            lufs_original=-60.0,
            lufs_target=-23.0,
            lufs_adjustment=0.0,
            peak_reduction_db=0.0,
            compression_ratio=1.0,
            hpf_cutoff=0.0,
            hum_frequencies=[],
            noise_type_detected="unknown",
            conditioning_features=torch.zeros(12)
        )
    
    def get_conditioning_report(self, features: ConditioningFeatures) -> str:
        """Generate a human-readable conditioning report."""
        report = f"""
Audio Conditioning Report:
==========================
Processing Applied:
  - Hum Filtering: {'Yes' if features.hum_filtered else 'No'}
  - High-Pass Filter: {'Yes' if features.hpf_applied else 'No'} (cutoff: {features.hpf_cutoff:.0f} Hz)
  - Denoising: {'Yes' if features.denoise_applied else 'No'}
  - Dereverberation: {'Yes' if features.dereverb_applied else 'No'}

Quality Metrics:
  - SNR Before: {features.snr_before:.1f} dB
  - SNR After: {features.snr_after:.1f} dB
  - Denoise Gain: {features.denoise_gain_db:.1f} dB
  - Estimated T60: {features.estimated_t60:.2f} s
  - Noise Type: {features.noise_type_detected}

Loudness Normalization:
  - Original LUFS: {features.lufs_original:.1f}
  - Target LUFS: {features.lufs_target:.1f}
  - LUFS Adjustment: {features.lufs_adjustment:.1f} dB
  - Peak Reduction: {features.peak_reduction_db:.1f} dB
  - Compression Ratio: {features.compression_ratio:.2f}

Hum Frequencies Detected: {features.hum_frequencies}
"""
        return report


# Utility function for easy integration
def create_audio_conditioning(sample_rate: int = 16000) -> AudioConditioningModule:
    """Factory function to create audio conditioning module with default settings."""
    return AudioConditioningModule(sample_rate=sample_rate)
