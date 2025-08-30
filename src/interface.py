import torch
import torchaudio
import numpy as np
import json
import argparse
import os
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import librosa
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

from models import AudioEncoder, TextEncoder, FusionLayer
from models.cross_attention import CrossModalAttention
from models.pooling import AttentiveStatsPooling
from models.classifier import AdvancedOpenMaxClassifier
from models.prototypes import PrototypeMemory
from models.quality_gates import FrontEndQualityGates, QualityMetrics
from data.preprocess import speed_perturb, add_noise_snr


class EmotionRecognitionInterface:
    """
    Comprehensive interface for Multilingual Multimodal Speech Emotion Recognition
    """
    
    def __init__(self, checkpoint_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        # Initialize models
        self._load_models()
        self._load_checkpoint()
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        
        # Emotion labels
        self.emotion_labels = ["Neutral", "Happy", "Sad", "Angry"]
        self.emotion_colors = ["gray", "yellow", "blue", "red"]
        
        print(f"✅ Emotion Recognition Interface loaded successfully on {device}")
        print(f"📁 Checkpoint: {checkpoint_path}")
    
    def _load_models(self):
        """Initialize all model components"""
        # Audio encoder with quality gates
        self.audio_encoder = AudioEncoder(
            model_name="facebook/wav2vec2-base",
            adapter_dim=256,
            freeze_base=True,
            use_quality_gates=True,
            vad_method="webrtc"
        ).to(self.device)
        
        # Text encoder
        self.text_encoder = TextEncoder(
            model_name="bert-base-multilingual-cased",
            hidden_size=768,
            num_layers=12
        ).to(self.device)
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(
            audio_dim=768,
            text_dim=768,
            shared_dim=256,
            num_heads=8
        ).to(self.device)
        
        # Pooling layers
        self.pool_a = AttentiveStatsPooling(768).to(self.device)
        self.pool_t = AttentiveStatsPooling(768).to(self.device)
        
        # Fusion layer
        self.fusion = FusionLayer(1536, 1536, 512).to(self.device)
        
        # Advanced classifier
        self.classifier = AdvancedOpenMaxClassifier(
            input_dim=512,
            num_labels=4,
            num_layers=35,
            base_dim=512,
            dropout=0.15
        ).to(self.device)
        
        # Prototype memory
        self.prototypes = PrototypeMemory(4, 512).to(self.device)
    
    def _load_checkpoint(self):
        """Load trained model weights"""
        print("🔄 Loading checkpoint...")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        self.audio_encoder.load_state_dict(checkpoint['audio_encoder'])
        self.text_encoder.load_state_dict(checkpoint['text_encoder'])
        self.cross_attention.load_state_dict(checkpoint['cross'])
        self.pool_a.load_state_dict(checkpoint['pool_a'])
        self.pool_t.load_state_dict(checkpoint['pool_t'])
        self.fusion.load_state_dict(checkpoint['fusion'])
        self.classifier.load_state_dict(checkpoint['classifier'])
        self.prototypes.load_state_dict(checkpoint['prototypes'])
        
        # Set to evaluation mode
        self.audio_encoder.eval()
        self.text_encoder.eval()
        self.cross_attention.eval()
        self.pool_a.eval()
        self.pool_t.eval()
        self.fusion.eval()
        self.classifier.eval()
        self.prototypes.eval()
        
        print("✅ Checkpoint loaded successfully!")
    
    def preprocess_audio(self, audio_path: str, target_sr: int = 16000) -> torch.Tensor:
        """Preprocess audio file for inference"""
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
        
        # Normalize
        waveform = waveform / torch.max(torch.abs(waveform))
        
        return waveform.squeeze()
    
    def preprocess_text(self, text: str, max_length: int = 512) -> torch.Tensor:
        """Preprocess text for inference"""
        # Tokenize
        tokens = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return tokens
    
    def predict_emotion(self, 
                       audio_path: Optional[str] = None,
                       text: Optional[str] = None,
                       use_tta: bool = True,
                       return_detailed: bool = True) -> Dict:
        """
        Predict emotion from audio and/or text
        
        Args:
            audio_path: Path to audio file
            text: Text transcript
            use_tta: Use test-time augmentation
            return_detailed: Return detailed analysis
            
        Returns:
            Dictionary with prediction results
        """
        
        with torch.no_grad():
            # Process audio if provided
            audio_features = None
            if audio_path:
                audio_waveform = self.preprocess_audio(audio_path)
                
                if use_tta:
                    # Test-time augmentation
                    augmented_audios = []
                    
                    # Original
                    augmented_audios.append(audio_waveform)
                    
                    # Speed perturbation
                    for factor in [0.9, 1.1]:
                        aug_audio = speed_perturb(audio_waveform, factor)
                        augmented_audios.append(aug_audio)
                    
                    # Noise addition
                    for snr in [15, 20]:
                        noisy_audio = add_noise_snr(audio_waveform, snr)
                        augmented_audios.append(noisy_audio)
                    
                    # Process all augmented versions
                    all_audio_features = []
                    for aug_audio in augmented_audios:
                        a_seq, a_mask = self.audio_encoder([aug_audio], [text] if text else None)
                        all_audio_features.append(a_seq)
                    
                    # Average the features
                    audio_features = torch.mean(torch.stack(all_audio_features), dim=0)
                else:
                    a_seq, a_mask = self.audio_encoder([audio_waveform], [text] if text else None)
                    audio_features = a_seq
            
            # Process text if provided
            text_features = None
            if text:
                text_tokens = self.preprocess_text(text)
                t_seq, t_mask = self.text_encoder([text_tokens])
                text_features = t_seq
            
            # Handle single modality
            if audio_features is None and text_features is None:
                raise ValueError("Either audio_path or text must be provided")
            
            if audio_features is None:
                # Text-only mode
                audio_features = torch.zeros_like(text_features)
                a_mask = torch.zeros_like(t_mask)
            elif text_features is None:
                # Audio-only mode
                text_features = torch.zeros_like(audio_features)
                t_mask = torch.zeros_like(a_mask)
            
            # Cross-modal attention
            a_enh, t_enh = self.cross_attention(audio_features, text_features, a_mask, t_mask)
            
            # Pooling
            a_vec = self.pool_a(a_enh, a_mask)
            t_vec = self.pool_t(t_enh, t_mask)
            
            # Fusion
            fused = self.fusion(a_vec, t_vec)
            
            # Classification
            logits, uncertainty, anchor_loss = self.classifier(
                fused, 
                use_openmax=True, 
                return_uncertainty=True
            )
            
            # Get predictions
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            confidence = 1 - uncertainty.squeeze()
            
            # Prepare results
            results = {
                'predictions': predictions.cpu().numpy(),
                'probabilities': probabilities.cpu().numpy(),
                'confidence': confidence.cpu().numpy(),
                'uncertainty': uncertainty.cpu().numpy(),
                'logits': logits.cpu().numpy(),
                'anchor_loss': anchor_loss.cpu().item(),
                'emotion_labels': [self.emotion_labels[pred] for pred in predictions.cpu().numpy()],
                'modalities': {
                    'audio': audio_path is not None,
                    'text': text is not None
                }
            }
            
            if return_detailed:
                results.update(self._get_detailed_analysis(logits, probabilities, uncertainty))
            
            return results
    
    def _get_detailed_analysis(self, logits: torch.Tensor, probabilities: torch.Tensor, 
                              uncertainty: torch.Tensor) -> Dict:
        """Get detailed analysis of predictions"""
        
        # Top-k predictions
        top_k = 2
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
        
        # Entropy (measure of uncertainty)
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)
        
        # Margin (difference between top-1 and top-2)
        margin = top_probs[:, 0] - top_probs[:, 1]
        
        # Calibration analysis
        calibration_error = torch.mean(torch.abs(probabilities.max(dim=1)[0] - (1 - uncertainty.squeeze())))
        
        return {
            'top_k_predictions': {
                'indices': top_indices.cpu().numpy(),
                'probabilities': top_probs.cpu().numpy(),
                'labels': [[self.emotion_labels[idx] for idx in batch] for batch in top_indices.cpu().numpy()]
            },
            'entropy': entropy.cpu().numpy(),
            'margin': margin.cpu().numpy(),
            'calibration_error': calibration_error.cpu().item(),
            'analysis': {
                'high_confidence': (1 - uncertainty.squeeze() > 0.8).cpu().numpy(),
                'low_confidence': (1 - uncertainty.squeeze() < 0.5).cpu().numpy(),
                'high_entropy': (entropy > 1.0).cpu().numpy(),
                'low_margin': (margin < 0.3).cpu().numpy()
            }
        }
    
    def visualize_results(self, results: Dict, save_path: Optional[str] = None):
        """Create comprehensive visualization of results"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Emotion Recognition Analysis', fontsize=16, fontweight='bold')
        
        # 1. Emotion probabilities bar chart
        ax1 = axes[0, 0]
        probs = results['probabilities'][0]  # First sample
        bars = ax1.bar(self.emotion_labels, probs, color=self.emotion_colors, alpha=0.7)
        ax1.set_title('Emotion Probabilities')
        ax1.set_ylabel('Probability')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, prob in zip(bars, probs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom')
        
        # 2. Confidence vs Uncertainty
        ax2 = axes[0, 1]
        confidence = results['confidence'][0]
        uncertainty = results['uncertainty'][0]
        ax2.bar(['Confidence', 'Uncertainty'], [confidence, uncertainty], 
                color=['green', 'red'], alpha=0.7)
        ax2.set_title('Confidence Analysis')
        ax2.set_ylabel('Score')
        ax2.set_ylim(0, 1)
        
        # 3. Logits comparison
        ax3 = axes[0, 2]
        logits = results['logits'][0]
        bars = ax3.bar(self.emotion_labels, logits, color=self.emotion_colors, alpha=0.7)
        ax3.set_title('Raw Logits')
        ax3.set_ylabel('Logit Score')
        
        # 4. Top-k predictions
        ax4 = axes[1, 0]
        top_k_results = results['top_k_predictions']
        top_labels = top_k_results['labels'][0]
        top_probs = top_k_results['probabilities'][0]
        
        bars = ax4.bar(range(len(top_labels)), top_probs, 
                      color=[self.emotion_colors[self.emotion_labels.index(label)] for label in top_labels],
                      alpha=0.7)
        ax4.set_title('Top-K Predictions')
        ax4.set_ylabel('Probability')
        ax4.set_xticks(range(len(top_labels)))
        ax4.set_xticklabels(top_labels, rotation=45)
        
        # 5. Analysis flags
        ax5 = axes[1, 1]
        analysis = results['analysis']
        flags = ['High Confidence', 'Low Confidence', 'High Entropy', 'Low Margin']
        values = [analysis['high_confidence'][0], analysis['low_confidence'][0], 
                 analysis['high_entropy'][0], analysis['low_margin'][0]]
        colors = ['green' if v else 'red' for v in values]
        
        bars = ax5.bar(flags, values, color=colors, alpha=0.7)
        ax5.set_title('Analysis Flags')
        ax5.set_ylabel('Flag Status')
        ax5.set_ylim(0, 1)
        ax5.set_yticks([0, 1])
        ax5.set_yticklabels(['False', 'True'])
        
        # 6. Summary statistics
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        summary_text = f"""
        Prediction Summary:
        
        Primary Emotion: {results['emotion_labels'][0]}
        Confidence: {results['confidence'][0]:.3f}
        Uncertainty: {results['uncertainty'][0]:.3f}
        Calibration Error: {results['calibration_error']:.3f}
        Anchor Loss: {results['anchor_loss']:.3f}
        
        Modalities Used:
        - Audio: {'✓' if results['modalities']['audio'] else '✗'}
        - Text: {'✓' if results['modalities']['text'] else '✗'}
        
        Analysis:
        - High Confidence: {'✓' if analysis['high_confidence'][0] else '✗'}
        - Low Confidence: {'✓' if analysis['low_confidence'][0] else '✗'}
        - High Entropy: {'✓' if analysis['high_entropy'][0] else '✗'}
        - Low Margin: {'✓' if analysis['low_margin'][0] else '✗'}
        """
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to: {save_path}")
        
        plt.show()
    
    def batch_predict(self, 
                     audio_paths: Optional[List[str]] = None,
                     texts: Optional[List[str]] = None,
                     use_tta: bool = True) -> List[Dict]:
        """Predict emotions for multiple samples"""
        
        if audio_paths is None and texts is None:
            raise ValueError("Either audio_paths or texts must be provided")
        
        results = []
        
        if audio_paths and texts:
            # Both modalities provided
            for audio_path, text in zip(audio_paths, texts):
                result = self.predict_emotion(audio_path=audio_path, text=text, use_tta=use_tta)
                results.append(result)
        elif audio_paths:
            # Audio only
            for audio_path in audio_paths:
                result = self.predict_emotion(audio_path=audio_path, use_tta=use_tta)
                results.append(result)
        else:
            # Text only
            for text in texts:
                result = self.predict_emotion(text=text, use_tta=use_tta)
                results.append(result)
        
        return results
    
    def save_results(self, results: Dict, output_path: str):
        """Save results to JSON file"""
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, dict):
                json_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        json_results[key][k] = v.tolist()
                    else:
                        json_results[key][k] = v
            else:
                json_results[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description="Emotion Recognition Interface")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--audio", help="Path to audio file")
    parser.add_argument("--text", help="Text transcript")
    parser.add_argument("--output", help="Output path for results JSON")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization")
    parser.add_argument("--save_viz", help="Path to save visualization")
    parser.add_argument("--no_tta", action="store_true", help="Disable test-time augmentation")
    parser.add_argument("--device", default="auto", help="Device to use (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Initialize interface
    interface = EmotionRecognitionInterface(args.checkpoint, device)
    
    # Make prediction
    results = interface.predict_emotion(
        audio_path=args.audio,
        text=args.text,
        use_tta=not args.no_tta,
        return_detailed=True
    )
    
    # Print results
    print("\n" + "="*50)
    print("🎭 EMOTION RECOGNITION RESULTS")
    print("="*50)
    print(f"Primary Emotion: {results['emotion_labels'][0]}")
    print(f"Confidence: {results['confidence'][0]:.3f}")
    print(f"Uncertainty: {results['uncertainty'][0]:.3f}")
    print(f"Calibration Error: {results['calibration_error']:.3f}")
    print(f"Anchor Loss: {results['anchor_loss']:.3f}")
    
    print("\n📊 Emotion Probabilities:")
    for emotion, prob in zip(results['emotion_labels'], results['probabilities'][0]):
        print(f"  {emotion}: {prob:.3f}")
    
    print("\n🔍 Top-K Predictions:")
    for i, (label, prob) in enumerate(zip(results['top_k_predictions']['labels'][0], 
                                         results['top_k_predictions']['probabilities'][0])):
        print(f"  {i+1}. {label}: {prob:.3f}")
    
    print("\n⚠️  Analysis Flags:")
    analysis = results['analysis']
    flags = ['High Confidence', 'Low Confidence', 'High Entropy', 'Low Margin']
    values = [analysis['high_confidence'][0], analysis['low_confidence'][0], 
             analysis['high_entropy'][0], analysis['low_margin'][0]]
    
    for flag, value in zip(flags, values):
        status = "✓" if value else "✗"
        print(f"  {flag}: {status}")
    
    # Save results
    if args.output:
        interface.save_results(results, args.output)
    
    # Visualize
    if args.visualize:
        interface.visualize_results(results, args.save_viz)


if __name__ == "__main__":
    main()


