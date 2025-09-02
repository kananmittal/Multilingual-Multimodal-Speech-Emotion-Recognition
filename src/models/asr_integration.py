
import torch
import torch.nn as nn
import numpy as np
import librosa
from typing import Dict, Tuple, Optional, List, Union
import logging
from dataclasses import dataclass
import warnings
import json
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import whisper
from sklearn.isotonic import IsotonicRegression
import re
from collections import defaultdict

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    warnings.warn("torchaudio not available. Some alignment features may be limited.")

try:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    WAV2VEC2_AVAILABLE = True
except ImportError:
    WAV2VEC2_AVAILABLE = False
    warnings.warn("wav2vec2 not available. Alignment will use simplified methods.")

@dataclass
class ASRResult:
    """Container for ASR results with confidence and alignment information."""
    # Basic ASR output
    text: str
    language: str
    detected_languages: List[str]
    
    # Confidence scores
    word_confidences: List[float]
    segment_confidence: float
    overall_confidence: float
    
    # Timestamps and alignment
    word_timestamps: List[Tuple[float, float]]  # (start_time, end_time)
    phone_alignment: List[Dict]  # Detailed phone-level alignment
    silence_regions: List[Tuple[float, float]]  # (start_time, end_time)
    
    # Code-switching information
    code_switches: List[Dict]  # Language switch points
    language_segments: List[Dict]  # Segments with language info
    
    # Processing features
    text_reliability_score: float
    attention_mask_weighted: torch.Tensor
    asr_features: torch.Tensor  # [8] tensor for fusion


class MultilingualASR:
    """Multilingual ASR with confidence estimation and code-switching awareness."""
    
    def __init__(self, model_name: str = "openai/whisper-large-v3", device: str = "cuda"):
        self.device = device
        self.model_name = model_name
        
        # Initialize Whisper model
        try:
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
            self.model.eval()
        except Exception as e:
            logging.warning(f"Could not load {model_name}, falling back to basic whisper")
            self.model = whisper.load_model("large-v3")
            self.processor = None
        
        # Confidence calibration
        self.confidence_calibrator = IsotonicRegression(out_of_bounds='clip')
        self.is_calibrated = False
        
        # Language detection for code-switching
        self.supported_languages = {
            'en': 'english', 'es': 'spanish', 'fr': 'french', 'de': 'german',
            'it': 'italian', 'pt': 'portuguese', 'ru': 'russian', 'ja': 'japanese',
            'ko': 'korean', 'zh': 'chinese', 'ar': 'arabic', 'hi': 'hindi'
        }
        
        logging.info(f"Multilingual ASR initialized with {model_name}")
    
    def transcribe_with_confidence(self, audio: np.ndarray, sample_rate: int = 16000) -> ASRResult:
        """
        Transcribe audio with confidence estimation and language detection.
        
        Args:
            audio: Audio waveform
            sample_rate: Audio sample rate
            
        Returns:
            ASRResult with transcription and confidence information
        """
        if self.processor is not None:
            return self._whisper_transformers_transcribe(audio, sample_rate)
        else:
            return self._whisper_basic_transcribe(audio, sample_rate)
    
    def _whisper_transformers_transcribe(self, audio: np.ndarray, sample_rate: int) -> ASRResult:
        """Transcribe using transformers Whisper with confidence estimation."""
        
        # Prepare input
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        
        # Tokenize input
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt").to(self.device)
        
        # Generate transcription
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs["input_features"],
                return_dict_in_generate=True,
                output_scores=True,
                max_length=448
            )
        
        # Decode transcription
        transcription = self.processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)[0]
        
        # Extract confidence scores from logits
        logits = torch.stack(generated_ids.scores, dim=1)
        probs = torch.softmax(logits, dim=-1)
        
        # Calculate word-level confidences
        word_confidences = self._extract_word_confidences(probs, generated_ids.sequences[0])
        
        # Detect languages and code-switching
        detected_languages, code_switches = self._detect_languages_and_switches(transcription)
        
        # Calculate segment confidence
        segment_confidence = np.mean(word_confidences) if word_confidences else 0.0
        
        # Apply confidence calibration if available
        if self.is_calibrated:
            segment_confidence = self.confidence_calibrator.predict([segment_confidence])[0]
        
        # Create ASR result
        result = ASRResult(
            text=transcription,
            language=detected_languages[0] if detected_languages else 'en',
            detected_languages=detected_languages,
            word_confidences=word_confidences,
            segment_confidence=segment_confidence,
            overall_confidence=segment_confidence,
            word_timestamps=[],  # Will be filled by alignment
            phone_alignment=[],
            silence_regions=[],
            code_switches=code_switches,
            language_segments=[],
            text_reliability_score=segment_confidence,
            attention_mask_weighted=torch.ones(len(word_confidences)),
            asr_features=torch.zeros(8)
        )
        
        return result
    
    def _whisper_basic_transcribe(self, audio: np.ndarray, sample_rate: int) -> ASRResult:
        """Transcribe using basic whisper library."""
        
        # Use basic whisper
        result = self.model.transcribe(audio, verbose=False)
        transcription = result["text"]
        
        # Estimate confidence (simplified)
        word_confidences = [0.8] * len(transcription.split())  # Placeholder
        segment_confidence = 0.8  # Placeholder
        
        # Detect languages
        detected_languages, code_switches = self._detect_languages_and_switches(transcription)
        
        # Create ASR result
        asr_result = ASRResult(
            text=transcription,
            language=detected_languages[0] if detected_languages else 'en',
            detected_languages=detected_languages,
            word_confidences=word_confidences,
            segment_confidence=segment_confidence,
            overall_confidence=segment_confidence,
            word_timestamps=[],
            phone_alignment=[],
            silence_regions=[],
            code_switches=code_switches,
            language_segments=[],
            text_reliability_score=segment_confidence,
            attention_mask_weighted=torch.ones(len(word_confidences)),
            asr_features=torch.zeros(8)
        )
        
        return asr_result
    
    def _extract_word_confidences(self, probs: torch.Tensor, tokens: torch.Tensor) -> List[float]:
        """Extract word-level confidence scores from token probabilities."""
        word_confidences = []
        
        # Get token probabilities
        token_probs = torch.max(probs, dim=-1)[0]
        
        # Group by words (simplified)
        current_word_probs = []
        for i, token_id in enumerate(tokens[1:]):  # Skip start token
            if token_id == self.processor.tokenizer.eos_token_id:
                break
            
            # Check if token is part of current word
            token_text = self.processor.tokenizer.decode([token_id])
            if token_text.startswith(' '):
                # New word
                if current_word_probs:
                    # Fix: ensure we're working with scalar values
                    current_probs = [p.item() if hasattr(p, 'item') else float(p) for p in current_word_probs]
                    word_confidences.append(float(np.mean(current_probs)))
                current_word_probs = [token_probs[i].item() if hasattr(token_probs[i], 'item') else float(token_probs[i])]
            else:
                # Continue current word
                current_word_probs.append(token_probs[i].item() if hasattr(token_probs[i], 'item') else float(token_probs[i]))
        
        # Add last word
        if current_word_probs:
            # Fix: ensure we're working with scalar values
            current_probs = [p.item() if hasattr(p, 'item') else float(p) for p in current_word_probs]
            word_confidences.append(float(np.mean(current_probs)))
        
        return word_confidences
    
    def _detect_languages_and_switches(self, text: str) -> Tuple[List[str], List[Dict]]:
        """Detect languages and code-switching in text."""
        detected_languages = []
        code_switches = []
        
        # Simple language detection based on character sets
        # In practice, use a proper language detection model
        
        # Check for different scripts
        if re.search(r'[а-яё]', text, re.IGNORECASE):
            detected_languages.append('ru')
        if re.search(r'[一-龯]', text):
            detected_languages.append('zh')
        if re.search(r'[あ-ん]', text):
            detected_languages.append('ja')
        if re.search(r'[가-힣]', text):
            detected_languages.append('ko')
        if re.search(r'[ا-ي]', text):
            detected_languages.append('ar')
        if re.search(r'[अ-ह]', text):
            detected_languages.append('hi')
        
        # Default to English if no specific script detected
        if not detected_languages:
            detected_languages.append('en')
        
        # Detect code switches (simplified)
        words = text.split()
        for i, word in enumerate(words):
            # Check for language-specific patterns
            if re.search(r'[а-яё]', word, re.IGNORECASE) and 'ru' not in detected_languages:
                code_switches.append({
                    'position': i,
                    'word': word,
                    'from_lang': detected_languages[0],
                    'to_lang': 'ru'
                })
        
        return detected_languages, code_switches
    
    def calibrate_confidence(self, validation_data: List[Tuple[np.ndarray, float]]):
        """Calibrate confidence scores using validation data."""
        predicted_confidences = []
        true_confidences = []
        
        for audio, true_conf in validation_data:
            result = self.transcribe_with_confidence(audio)
            predicted_confidences.append(result.segment_confidence)
            true_confidences.append(true_conf)
        
        # Fit isotonic regression
        self.confidence_calibrator.fit(predicted_confidences, true_confidences)
        self.is_calibrated = True
        
        logging.info("ASR confidence calibration completed")


class TimestampAlignment:
    """Timestamp alignment for ASR output with audio."""
    
    def __init__(self, device: str = "cuda"):
        # Ensure device is valid
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        
        self.device = device
        
        # Initialize wav2vec2 for alignment
        if WAV2VEC2_AVAILABLE:
            try:
                self.alignment_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base").to(device)
                self.alignment_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            except Exception as e:
                logging.warning(f"Failed to initialize wav2vec2 alignment: {e}")
                self.alignment_model = None
                self.alignment_processor = None
        else:
            self.alignment_model = None
            self.alignment_processor = None
        
        logging.info("Timestamp alignment initialized")
    
    def align_transcription(self, audio: np.ndarray, asr_result: ASRResult, 
                          sample_rate: int = 16000) -> ASRResult:
        """
        Align ASR transcription with audio timestamps.
        
        Args:
            audio: Audio waveform
            asr_result: ASR result to align
            sample_rate: Audio sample rate
            
        Returns:
            ASRResult with aligned timestamps
        """
        if self.alignment_model is not None:
            return self._wav2vec2_alignment(audio, asr_result, sample_rate)
        else:
            return self._simple_alignment(audio, asr_result, sample_rate)
    
    def _wav2vec2_alignment(self, audio: np.ndarray, asr_result: ASRResult, 
                           sample_rate: int) -> ASRResult:
        """Align using wav2vec2-based forced alignment."""
        
        # Prepare audio
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        
        # Process audio
        inputs = self.alignment_processor(audio, sampling_rate=16000, return_tensors="pt").to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.alignment_model(**inputs)
            logits = outputs.logits
        
        # Extract timestamps (simplified implementation)
        # In practice, use proper forced alignment algorithms
        word_timestamps = self._extract_word_timestamps_simple(audio, asr_result.text)
        phone_alignment = self._extract_phone_alignment_simple(audio, asr_result.text)
        silence_regions = self._detect_silence_regions(audio)
        
        # Update ASR result
        asr_result.word_timestamps = word_timestamps
        asr_result.phone_alignment = phone_alignment
        asr_result.silence_regions = silence_regions
        
        return asr_result
    
    def _simple_alignment(self, audio: np.ndarray, asr_result: ASRResult, 
                         sample_rate: int) -> ASRResult:
        """Simple alignment using energy-based segmentation."""
        
        # Simple word-level alignment based on energy
        word_timestamps = self._extract_word_timestamps_simple(audio, asr_result.text)
        phone_alignment = self._extract_phone_alignment_simple(audio, asr_result.text)
        silence_regions = self._detect_silence_regions(audio)
        
        # Update ASR result
        asr_result.word_timestamps = word_timestamps
        asr_result.phone_alignment = phone_alignment
        asr_result.silence_regions = silence_regions
        
        return asr_result
    
    def _extract_word_timestamps_simple(self, audio: np.ndarray, text: str) -> List[Tuple[float, float]]:
        """Extract word timestamps using simple energy-based segmentation."""
        words = text.split()
        word_timestamps = []
        
        # Calculate energy envelope
        frame_length = 1024
        hop_length = 512
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Find speech segments
        threshold = np.percentile(energy, 30)
        speech_frames = energy > threshold
        
        # Simple word segmentation
        word_duration = len(audio) / len(words) if words else 0
        
        for i, word in enumerate(words):
            start_time = i * word_duration
            end_time = (i + 1) * word_duration
            word_timestamps.append((start_time, end_time))
        
        return word_timestamps
    
    def _extract_phone_alignment_simple(self, audio: np.ndarray, text: str) -> List[Dict]:
        """Extract phone-level alignment (simplified)."""
        phone_alignment = []
        
        # Simplified phone alignment
        # In practice, use proper phonetic alignment
        words = text.split()
        for word in words:
            # Estimate phone boundaries (simplified)
            phone_count = len(word)  # Rough estimate
            for j in range(phone_count):
                phone_alignment.append({
                    'phone': word[j] if j < len(word) else '',
                    'start_time': 0.0,  # Placeholder
                    'end_time': 0.1,    # Placeholder
                    'confidence': 0.8   # Placeholder
                })
        
        return phone_alignment
    
    def _detect_silence_regions(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """Detect silence regions in audio."""
        silence_regions = []
        
        # Calculate energy
        frame_length = 1024
        hop_length = 512
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Find silence regions
        threshold = np.percentile(energy, 20)
        silence_frames = energy <= threshold
        
        # Convert frame indices to time
        frame_times = librosa.frames_to_time(np.arange(len(energy)), sr=16000, hop_length=hop_length)
        
        # Find continuous silence regions
        in_silence = False
        silence_start = 0
        
        for i, is_silence in enumerate(silence_frames):
            if is_silence and not in_silence:
                silence_start = frame_times[i]
                in_silence = True
            elif not is_silence and in_silence:
                silence_end = frame_times[i]
                if silence_end - silence_start > 0.1:  # Minimum 100ms
                    silence_regions.append((silence_start, silence_end))
                in_silence = False
        
        return silence_regions


class ConfidenceAwareTextProcessor:
    """Process text with ASR confidence-aware weighting."""
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.low_confidence_threshold = 0.6
        self.very_low_confidence_threshold = 0.3
        
        logging.info("Confidence-aware text processor initialized")
    
    def process_text_with_confidence(self, asr_result: ASRResult) -> ASRResult:
        """
        Process text with confidence-aware weighting.
        
        Args:
            asr_result: ASR result with confidence information
            
        Returns:
            Updated ASR result with confidence-aware features
        """
        # Calculate text reliability score
        text_reliability_score = np.mean(asr_result.word_confidences) if asr_result.word_confidences else 0.0
        
        # Create confidence-aware attention mask
        attention_mask_weighted = self._create_confidence_aware_mask(asr_result.word_confidences)
        
        # Update ASR result
        asr_result.text_reliability_score = text_reliability_score
        asr_result.attention_mask_weighted = attention_mask_weighted
        
        return asr_result
    
    def _create_confidence_aware_mask(self, word_confidences: List[float]) -> torch.Tensor:
        """Create attention mask weighted by ASR confidence."""
        if not word_confidences:
            return torch.ones(1)
        
        mask_weights = []
        for conf in word_confidences:
            if conf < self.very_low_confidence_threshold:
                # Very low confidence - mask out
                mask_weights.append(0.0)
            elif conf < self.low_confidence_threshold:
                # Low confidence - reduce weight
                mask_weights.append(0.3)
            else:
                # High confidence - full weight
                mask_weights.append(1.0)
        
        return torch.tensor(mask_weights, dtype=torch.float32)


class EnhancedASRIntegration(nn.Module):
    """
    Enhanced ASR Integration and Alignment Module.
    
    Provides multilingual ASR with confidence estimation, timestamp alignment,
    and code-switching awareness.
    """
    
    def __init__(self, model_name: str = "openai/whisper-large-v3", device: str = "cuda"):
        super().__init__()
        
        self.device = device
        
        # Initialize components
        self.asr = MultilingualASR(model_name, device)
        self.alignment = TimestampAlignment(device)
        self.text_processor = ConfidenceAwareTextProcessor()
        
        # ASR feature projection for downstream fusion
        self.asr_projection = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 8)
        )
        
        logging.info(f"Enhanced ASR Integration initialized with {model_name}")
    
    def forward(self, audio: torch.Tensor, sample_rate: int = 16000) -> ASRResult:
        """
        Process audio through enhanced ASR pipeline.
        
        Args:
            audio: Audio tensor [batch_size, samples] or [samples]
            sample_rate: Audio sample rate
            
        Returns:
            ASRResult with transcription, confidence, and alignment
        """
        # Convert to numpy
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy()
        else:
            audio_np = audio
        
        # Handle batch dimension
        if audio_np.ndim == 2:
            # Process first sample in batch
            audio_sample = audio_np[0]
        else:
            audio_sample = audio_np
        
        # ASR transcription with confidence
        asr_result = self.asr.transcribe_with_confidence(audio_sample, sample_rate)
        
        # Timestamp alignment
        asr_result = self.alignment.align_transcription(audio_sample, asr_result, sample_rate)
        
        # Confidence-aware text processing
        asr_result = self.text_processor.process_text_with_confidence(asr_result)
        
        # Create ASR features for fusion
        asr_features = self._create_asr_features(asr_result)
        asr_result.asr_features = asr_features
        
        return asr_result
    
    def _create_asr_features(self, asr_result: ASRResult) -> torch.Tensor:
        """Create ASR features for downstream fusion."""
        features = torch.tensor([
            asr_result.overall_confidence,
            asr_result.text_reliability_score,
            len(asr_result.detected_languages) / 5.0,  # Normalize language count
            len(asr_result.code_switches) / 10.0,      # Normalize code switches
            len(asr_result.silence_regions) / 20.0,    # Normalize silence regions
            np.mean([end - start for start, end in asr_result.word_timestamps]) if asr_result.word_timestamps else 0.0,
            len(asr_result.text.split()) / 50.0,       # Normalize word count
            float(len(asr_result.text) > 0)            # Has transcription
        ], dtype=torch.float32)
        
        # Project features
        features = self.asr_projection(features.unsqueeze(0)).squeeze(0)
        
        return features
    
    def get_asr_report(self, asr_result: ASRResult) -> str:
        """Generate a human-readable ASR report."""
        report = f"""
Enhanced ASR Report:
===================
Transcription:
  Text: "{asr_result.text}"
  Language: {asr_result.language}
  Detected Languages: {asr_result.detected_languages}

Confidence Analysis:
  Overall Confidence: {asr_result.overall_confidence:.3f}
  Text Reliability Score: {asr_result.text_reliability_score:.3f}
  Word Confidences: {asr_result.word_confidences[:5]}... (showing first 5)

Alignment Information:
  Word Timestamps: {len(asr_result.word_timestamps)} words aligned
  Silence Regions: {len(asr_result.silence_regions)} regions detected
  Phone Alignment: {len(asr_result.phone_alignment)} phones aligned

Code-Switching Analysis:
  Code Switches: {len(asr_result.code_switches)} detected
  Switch Details: {asr_result.code_switches[:3]}... (showing first 3)

Processing Features:
  Attention Mask Weighted: {asr_result.attention_mask_weighted.shape}
  ASR Features: {asr_result.asr_features.shape}
"""
        return report


# Utility function for easy integration
def create_enhanced_asr(model_name: str = "openai/whisper-large-v3", 
                       device: str = "cuda") -> EnhancedASRIntegration:
    """Factory function to create enhanced ASR integration with default settings."""
    return EnhancedASRIntegration(model_name, device)
