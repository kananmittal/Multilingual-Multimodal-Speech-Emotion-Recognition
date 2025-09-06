import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from typing import Dict, Tuple, Optional, List
import logging
from dataclasses import dataclass
import warnings

try:
    import webrtcvad
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    warnings.warn("webrtcvad not available. Using librosa-based VAD.")

try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    warnings.warn("langdetect not available. Language detection will be disabled.")

@dataclass
class QualityMetrics:
    """Container for all quality assessment metrics."""
    # VAD metrics
    speech_prob: float
    speech_segments: List[Tuple[float, float]]  # (start_time, end_time)
    
    # Signal quality metrics
    snr_db: float
    clipping_percent: float
    spectral_naturalness: float
    
    # Language metrics
    lid_entropy: float
    dominant_language: str
    dominant_language_conf: float
    
    # Content type metrics
    music_prob: float
    laughter_prob: float
    
    # Decision metrics
    abstain_recommendation: str  # 'reject', 'uncertain', 'accept'
    quality_score: float  # Overall quality score [0, 1]
    
    # Additional features for downstream fusion
    quality_features: torch.Tensor  # [8] tensor for fusion


class VoiceActivityDetector:
    """Voice Activity Detection using WebRTC or librosa-based approach."""
    
    def __init__(self, method: str = "webrtc", sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.method = method
        
        if method == "webrtc" and WEBRTC_AVAILABLE:
            self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2
            self.frame_duration = 30  # ms
            self.frame_size = int(sample_rate * self.frame_duration / 1000)
        elif method == "librosa":
            self.frame_duration = 25  # ms
            self.frame_size = int(sample_rate * 0.025)  # 25ms windows
            self.hop_size = int(sample_rate * 0.010)    # 10ms hop
        else:
            raise ValueError(f"VAD method '{method}' not available")
    
    def detect_speech(self, audio: np.ndarray) -> Tuple[float, List[Tuple[float, float]]]:
        """
        Detect speech activity in audio.
        
        Args:
            audio: Audio waveform [samples]
            
        Returns:
            speech_prob: Overall speech probability [0, 1]
            speech_segments: List of (start_time, end_time) tuples
        """
        if self.method == "webrtc":
            return self._webrtc_vad(audio)
        else:
            return self._librosa_vad(audio)
    
    def _webrtc_vad(self, audio: np.ndarray) -> Tuple[float, List[Tuple[float, float]]]:
        """WebRTC-based VAD implementation."""
        # Ensure audio is 16-bit PCM
        if audio.dtype != np.int16:
            audio = (audio * 32767).astype(np.int16)
        
        # Process in frames
        frame_size = self.frame_size
        speech_frames = []
        
        for i in range(0, len(audio) - frame_size + 1, frame_size):
            frame = audio[i:i + frame_size]
            if len(frame) == frame_size:
                is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)
                speech_frames.append(is_speech)
        
        # Convert to speech segments
        speech_prob = np.mean(speech_frames) if speech_frames else 0.0
        speech_segments = self._frames_to_segments(speech_frames)
        
        return speech_prob, speech_segments
    
    def _librosa_vad(self, audio: np.ndarray) -> Tuple[float, List[Tuple[float, float]]]:
        """Librosa-based energy VAD implementation."""
        # Compute energy
        energy = librosa.feature.rms(
            y=audio, 
            frame_length=self.frame_size, 
            hop_length=self.hop_size
        )[0]
        
        # Dynamic threshold based on energy distribution
        threshold = np.percentile(energy, 30) + 0.1 * np.std(energy)
        
        # Detect speech frames
        speech_frames = energy > threshold
        
        # Apply smoothing
        speech_frames = self._smooth_speech_frames(speech_frames)
        
        # Convert to segments
        speech_prob = np.mean(speech_frames)
        speech_segments = self._frames_to_segments(speech_frames)
        
        return speech_prob, speech_segments
    
    def _smooth_speech_frames(self, speech_frames: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Apply median smoothing to reduce noise in VAD output."""
        from scipy.ndimage import median_filter
        return median_filter(speech_frames, size=window_size)
    
    def _frames_to_segments(self, speech_frames: np.ndarray) -> List[Tuple[float, float]]:
        """Convert frame-level speech detection to time segments."""
        if len(speech_frames) == 0:
            return []
        
        segments = []
        in_speech = False
        start_time = 0
        
        for i, is_speech in enumerate(speech_frames):
            if bool(is_speech) and not in_speech:
                start_time = i * self.frame_duration / 1000.0
                in_speech = True
            elif not bool(is_speech) and in_speech:
                end_time = i * self.frame_duration / 1000.0
                segments.append((start_time, end_time))
                in_speech = False
        
        # Handle case where speech continues to end
        if in_speech:
            end_time = len(speech_frames) * self.frame_duration / 1000.0
            segments.append((start_time, end_time))
        
        return segments


class SignalQualityAssessor:
    """Assess signal quality including SNR, clipping, and spectral characteristics."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def assess_quality(self, audio: np.ndarray) -> Tuple[float, float, float]:
        """
        Assess signal quality.
        
        Args:
            audio: Audio waveform [samples]
            
        Returns:
            snr_db: Signal-to-noise ratio in dB
            clipping_percent: Percentage of clipped samples
            spectral_naturalness: Spectral naturalness score [0, 1]
        """
        snr_db = self._estimate_snr(audio)
        clipping_percent = self._detect_clipping(audio)
        spectral_naturalness = self._compute_spectral_naturalness(audio)
        
        return snr_db, clipping_percent, spectral_naturalness
    
    def _estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate SNR using spectral subtraction method."""
        # Compute spectrogram
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        
        # Estimate noise from first and last 10% of frames
        noise_frames = int(0.1 * magnitude.shape[1])
        noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1)
        noise_spectrum = np.mean(magnitude[:, -noise_frames:], axis=1)
        noise_spectrum = (noise_spectrum + noise_spectrum) / 2
        
        # Compute signal spectrum (excluding noise frames)
        signal_spectrum = np.mean(magnitude[:, noise_frames:-noise_frames], axis=1)
        
        # Compute SNR
        signal_power = np.mean(signal_spectrum ** 2)
        noise_power = np.mean(noise_spectrum ** 2)
        
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = 50.0  # Very high SNR if no noise detected
        
        return max(0.0, min(50.0, snr_db))  # Clip to reasonable range
    
    def _detect_clipping(self, audio: np.ndarray) -> float:
        """Detect percentage of clipped samples."""
        # Normalize audio to [-1, 1] range
        audio_norm = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
        
        # Count samples near the extremes
        clip_threshold = 0.95
        clipped_samples = np.sum(np.abs(audio_norm) > clip_threshold)
        clipping_percent = (clipped_samples / len(audio)) * 100
        
        return clipping_percent
    
    def _compute_spectral_naturalness(self, audio: np.ndarray) -> float:
        """Compute spectral naturalness score based on spectral characteristics."""
        # Compute spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
        
        # Compute naturalness score based on typical speech characteristics
        # Speech typically has centroid around 1000-3000 Hz
        centroid_score = 1.0 - np.clip(np.abs(np.mean(spectral_centroid) - 2000) / 2000, 0, 1)
        
        # Speech has moderate rolloff (around 0.85)
        rolloff_score = 1.0 - np.clip(np.abs(np.mean(spectral_rolloff) - 0.85) / 0.15, 0, 1)
        
        # Speech has moderate bandwidth
        bandwidth_score = 1.0 - np.clip(np.abs(np.mean(spectral_bandwidth) - 1000) / 1000, 0, 1)
        
        # Combine scores
        naturalness = (centroid_score + rolloff_score + bandwidth_score) / 3
        return float(naturalness)


class LanguageIdentifier:
    """Language identification with entropy calculation."""
    
    def __init__(self):
        self.available = LANGDETECT_AVAILABLE
        if not self.available:
            logging.warning("Language detection not available. Using fallback.")
    
    def identify_language(self, text: str) -> Tuple[float, str, float]:
        """
        Identify language and compute entropy.
        
        Args:
            text: Input text
            
        Returns:
            lid_entropy: Language identification entropy
            dominant_language: Most likely language code
            dominant_language_conf: Confidence in dominant language
        """
        if not self.available or not text.strip():
            return 1.5, "unknown", 0.0  # High entropy for unknown
        
        try:
            # Get language probabilities (simulated since langdetect doesn't provide probabilities)
            # In practice, you'd use a model that provides probability distributions
            detected_lang = detect(text)
            
            # Simulate probability distribution (in real implementation, use proper LID model)
            # This is a simplified version - replace with actual LID model
            languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh']
            if detected_lang in languages:
                # Simulate high confidence for detected language
                probs = [0.05] * len(languages)  # Base probability
                lang_idx = languages.index(detected_lang)
                probs[lang_idx] = 0.7  # High probability for detected language
                
                # Normalize
                probs = np.array(probs) / np.sum(probs)
            else:
                # Uniform distribution for unknown language
                probs = np.ones(len(languages)) / len(languages)
            
            # Compute entropy
            lid_entropy = -np.sum(probs * np.log(probs + 1e-10))
            
            # Get dominant language
            dominant_idx = np.argmax(probs)
            dominant_language = languages[dominant_idx]
            dominant_language_conf = probs[dominant_idx]
            
            return float(lid_entropy), dominant_language, float(dominant_language_conf)
            
        except Exception as e:
            logging.warning(f"Language detection failed: {e}")
            return 1.5, "unknown", 0.0


class ContentTypeDetector:
    """Detect music, laughter, and other non-speech content."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        # Simple rule-based detector (in practice, use trained classifier)
        self.mfcc_extractor = lambda x: librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=13)
    
    def detect_content_type(self, audio: np.ndarray) -> Tuple[float, float]:
        """
        Detect music and laughter probability.
        
        Args:
            audio: Audio waveform [samples]
            
        Returns:
            music_prob: Music probability [0, 1]
            laughter_prob: Laughter probability [0, 1]
        """
        # Extract MFCC features
        mfcc = self.mfcc_extractor(audio)
        
        # Simple rule-based detection (replace with trained classifier)
        # Music detection based on spectral characteristics
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        
        # Music typically has higher spectral centroid and rolloff
        music_score = np.mean(spectral_centroid) / 4000  # Normalize
        music_score = np.clip(music_score, 0, 1)
        
        # Laughter detection based on periodicity and energy
        # Laughter has characteristic periodic patterns
        energy = librosa.feature.rms(y=audio)[0]
        energy_variance = np.var(energy)
        laughter_score = np.clip(energy_variance / 0.1, 0, 1)  # Normalize
        
        return float(music_score), float(laughter_score)


class EarlyAbstainPolicy:
    """Early abstain policy for quality-based rejection."""
    
    def __init__(self):
        # Thresholds for decision making
        self.snr_threshold_low = 5.0   # dB
        self.snr_threshold_high = 10.0 # dB
        self.clipping_threshold = 30.0  # %
        self.speech_prob_threshold_low = 0.4
        self.speech_prob_threshold_high = 0.8
        self.lid_entropy_threshold = 1.5
        self.music_prob_threshold = 0.2
    
    def make_decision(self, metrics: QualityMetrics) -> str:
        """
        Make early abstain decision.
        
        Args:
            metrics: Quality metrics
            
        Returns:
            decision: 'reject', 'uncertain', or 'accept'
        """
        # Reject conditions
        if (metrics.snr_db < self.snr_threshold_low or 
            metrics.clipping_percent > self.clipping_threshold or
            metrics.speech_prob < self.speech_prob_threshold_low):
            return 'reject'
        
        # Uncertain conditions
        if (self.snr_threshold_low <= metrics.snr_db < self.snr_threshold_high or
            metrics.lid_entropy > self.lid_entropy_threshold or
            metrics.music_prob > self.music_prob_threshold):
            return 'uncertain'
        
        # Accept conditions
        if (metrics.snr_db >= self.snr_threshold_high and
            metrics.speech_prob >= self.speech_prob_threshold_high and
            metrics.lid_entropy < self.lid_entropy_threshold):
            return 'accept'
        
        # Default to uncertain
        return 'uncertain'
    
    def compute_quality_score(self, metrics: QualityMetrics) -> float:
        """Compute overall quality score [0, 1]."""
        # Normalize individual metrics
        snr_score = np.clip(metrics.snr_db / 20.0, 0, 1)  # 20dB is excellent
        speech_score = metrics.speech_prob
        clipping_score = 1.0 - np.clip(metrics.clipping_percent / 100.0, 0, 1)
        naturalness_score = metrics.spectral_naturalness
        lid_score = 1.0 - np.clip(metrics.lid_entropy / 2.0, 0, 1)  # Lower entropy is better
        music_score = 1.0 - metrics.music_prob  # Less music is better
        
        # Weighted combination
        quality_score = (
            0.25 * snr_score +
            0.25 * speech_score +
            0.15 * clipping_score +
            0.15 * naturalness_score +
            0.10 * lid_score +
            0.10 * music_score
        )
        
        return float(quality_score)


class FrontEndQualityGates(nn.Module):
    """
    Front-End Quality Gates Module.
    
    Multi-stage quality assessment and filtering system that processes audio
    before feature extraction to improve efficiency and quality.
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 vad_method: str = "webrtc",
                 enable_language_detection: bool = True):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.enable_language_detection = enable_language_detection
        
        # Initialize quality assessment modules
        self.vad = VoiceActivityDetector(method=vad_method, sample_rate=sample_rate)
        self.quality_assessor = SignalQualityAssessor(sample_rate=sample_rate)
        self.language_identifier = LanguageIdentifier()
        self.content_detector = ContentTypeDetector(sample_rate=sample_rate)
        self.abstain_policy = EarlyAbstainPolicy()
        
        # Quality feature projection for downstream fusion
        self.quality_projection = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 8)
        )
        
        logging.info(f"Front-End Quality Gates initialized with VAD method: {vad_method}")
    
    def forward(self, 
                audio: torch.Tensor, 
                text: Optional[str] = None) -> Tuple[torch.Tensor, QualityMetrics, bool]:
        """
        Process audio through quality gates.
        
        Args:
            audio: Audio tensor [batch_size, samples] or [samples]
            text: Optional text for language detection
            
        Returns:
            processed_audio: Quality-filtered audio tensor
            metrics: Quality metrics
            should_process: Boolean indicating if audio should be processed further
        """
        # Ensure audio is numpy array
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy()
        else:
            audio_np = audio
        
        # Handle batch dimension
        if audio_np.ndim == 2:
            # Process each sample in batch
            processed_audios = []
            all_metrics = []
            should_process_batch = True
            
            for i in range(audio_np.shape[0]):
                audio_sample = audio_np[i]
                text_sample = text[i] if isinstance(text, list) and i < len(text) else text
                
                processed_audio, metrics, should_process = self._process_single_sample(
                    audio_sample, text_sample
                )
                
                processed_audios.append(processed_audio)
                all_metrics.append(metrics)
                should_process_batch = should_process_batch and should_process
            
            # Stack results
            processed_audio = torch.stack(processed_audios) if processed_audios else audio
            metrics = all_metrics[0] if all_metrics else self._create_empty_metrics()
            
            return processed_audio, metrics, should_process_batch
        else:
            # Single sample
            return self._process_single_sample(audio_np, text)
    
    def _process_single_sample(self, 
                              audio: np.ndarray, 
                              text: Optional[str]) -> Tuple[torch.Tensor, QualityMetrics, bool]:
        """Process a single audio sample through quality gates."""
        
        # 1. Voice Activity Detection
        speech_prob, speech_segments = self.vad.detect_speech(audio)
        
        # 2. Signal Quality Assessment
        snr_db, clipping_percent, spectral_naturalness = self.quality_assessor.assess_quality(audio)
        
        # 3. Language Identification (if text provided and enabled)
        if text and self.enable_language_detection:
            lid_entropy, dominant_language, dominant_language_conf = self.language_identifier.identify_language(text)
        else:
            lid_entropy, dominant_language, dominant_language_conf = 1.0, "unknown", 0.0
        
        # 4. Content Type Detection
        music_prob, laughter_prob = self.content_detector.detect_content_type(audio)
        
        # 5. Create quality metrics
        metrics = QualityMetrics(
            speech_prob=speech_prob,
            speech_segments=speech_segments,
            snr_db=snr_db,
            clipping_percent=clipping_percent,
            spectral_naturalness=spectral_naturalness,
            lid_entropy=lid_entropy,
            dominant_language=dominant_language,
            dominant_language_conf=dominant_language_conf,
            music_prob=music_prob,
            laughter_prob=laughter_prob,
            abstain_recommendation="",  # Will be set below
            quality_score=0.0,  # Will be set below
            quality_features=torch.zeros(8)  # Will be set below
        )
        
        # 6. Early Abstain Decision
        abstain_decision = self.abstain_policy.make_decision(metrics)
        metrics.abstain_recommendation = abstain_decision
        
        # 7. Compute quality score
        quality_score = self.abstain_policy.compute_quality_score(metrics)
        metrics.quality_score = quality_score
        
        # 8. Create quality features for downstream fusion
        device = next(self.quality_projection.parameters()).device
        quality_features = torch.tensor([
            speech_prob,
            snr_db / 50.0,  # Normalize to [0, 1]
            clipping_percent / 100.0,
            spectral_naturalness,
            lid_entropy / 2.0,  # Normalize to [0, 1]
            dominant_language_conf,
            music_prob,
            laughter_prob
        ], dtype=torch.float32, device=device)
        quality_features = self.quality_projection(quality_features.unsqueeze(0)).squeeze(0)
        metrics.quality_features = quality_features
        
        # 9. Decision logic
        should_process = abstain_decision == 'accept'
        
        # 10. Return processed audio (apply quality-based filtering if needed)
        processed_audio = torch.tensor(audio, dtype=torch.float32, device=device)
        
        # If rejected, return zero audio (or could return None)
        if abstain_decision == 'reject':
            processed_audio = torch.zeros_like(processed_audio)
        
        return processed_audio, metrics, should_process
    
    def _create_empty_metrics(self) -> QualityMetrics:
        """Create empty quality metrics for error cases."""
        return QualityMetrics(
            speech_prob=0.0,
            speech_segments=[],
            snr_db=0.0,
            clipping_percent=100.0,
            spectral_naturalness=0.0,
            lid_entropy=2.0,
            dominant_language="unknown",
            dominant_language_conf=0.0,
            music_prob=1.0,
            laughter_prob=0.0,
            abstain_recommendation="reject",
            quality_score=0.0,
            quality_features=torch.zeros(8)
        )
    
    def get_quality_report(self, metrics: QualityMetrics) -> str:
        """Generate a human-readable quality report."""
        report = f"""
Quality Assessment Report:
==========================
Speech Activity:
  - Speech Probability: {metrics.speech_prob:.3f}
  - Speech Segments: {len(metrics.speech_segments)} segments

Signal Quality:
  - SNR: {metrics.snr_db:.1f} dB
  - Clipping: {metrics.clipping_percent:.1f}%
  - Spectral Naturalness: {metrics.spectral_naturalness:.3f}

Language:
  - Dominant Language: {metrics.dominant_language}
  - Language Confidence: {metrics.dominant_language_conf:.3f}
  - Language Entropy: {metrics.lid_entropy:.3f}

Content Type:
  - Music Probability: {metrics.music_prob:.3f}
  - Laughter Probability: {metrics.laughter_prob:.3f}

Decision:
  - Recommendation: {metrics.abstain_recommendation.upper()}
  - Quality Score: {metrics.quality_score:.3f}
"""
        return report


# Utility function for easy integration
def create_quality_gates(sample_rate: int = 16000, 
                        vad_method: str = "webrtc",
                        enable_language_detection: bool = True) -> FrontEndQualityGates:
    """Factory function to create quality gates with default settings."""
    return FrontEndQualityGates(
        sample_rate=sample_rate,
        vad_method=vad_method,
        enable_language_detection=enable_language_detection
    )
