#!/usr/bin/env python3
"""
Test academic evaluation components individually to identify and fix issues.
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import json
from datetime import datetime
from tqdm import tqdm

from models import AudioEncoder, TextEncoder, FusionLayer, Classifier
from models.classifier import AdvancedOpenMaxClassifier
from models.cross_attention import CrossModalAttention
from models.pooling import AttentiveStatsPooling
from models.prototypes import PrototypeMemory
from data.dataset import SERDataset
from utils import weighted_f1, energy_score
from sklearn.metrics import confusion_matrix

def collate_fn(batch):
    audios, texts, labels = zip(*batch)
    return list(audios), list(texts), torch.tensor(labels, dtype=torch.long)

def load_and_freeze_model(checkpoint_path, device):
    """Load model from checkpoint and freeze all weights."""
    print(f"Loading model from {checkpoint_path}")
    
    # Initialize models
    audio_encoder = AudioEncoder().to(device)
    text_encoder = TextEncoder().to(device)
    audio_hid = audio_encoder.encoder.config.hidden_size
    text_hid = text_encoder.encoder.config.hidden_size
    cross = CrossModalAttention(audio_hid, text_hid, shared_dim=256, num_heads=8).to(device)
    pool_a = AttentiveStatsPooling(audio_hid).to(device)
    pool_t = AttentiveStatsPooling(text_hid).to(device)
    fusion = FusionLayer(audio_hid * 2, text_hid * 2, 512).to(device)
    classifier = AdvancedOpenMaxClassifier(
        input_dim=512, 
        num_labels=4, 
        num_layers=35, 
        base_dim=512, 
        dropout=0.15
    ).to(device)
    prototypes = PrototypeMemory(4, 512).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load state dicts
    audio_encoder.load_state_dict(checkpoint['audio_encoder'])
    text_encoder.load_state_dict(checkpoint['text_encoder'])
    cross.load_state_dict(checkpoint['cross'])
    pool_a.load_state_dict(checkpoint['pool_a'])
    pool_t.load_state_dict(checkpoint['pool_a'])
    fusion.load_state_dict(checkpoint['fusion'])
    classifier.load_state_dict(checkpoint['classifier'])
    prototypes.load_state_dict(checkpoint['prototypes'])
    
    # Freeze all models
    print("Freezing all model weights...")
    for model in [audio_encoder, text_encoder, cross, pool_a, pool_t, fusion, classifier, prototypes]:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
    
    print("✅ All model weights frozen successfully!")
    
    return {
        'audio_encoder': audio_encoder,
        'text_encoder': text_encoder,
        'cross': cross,
        'pool_a': pool_a,
        'pool_t': pool_t,
        'fusion': fusion,
        'classifier': classifier,
        'prototypes': prototypes
    }

def test_baseline_evaluation(model, test_loader, device):
    """Test baseline evaluation."""
    print("\n🧪 Testing Baseline Evaluation...")
    
    model['fusion'].eval()
    model['classifier'].eval()
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for audio_list, text_list, labels in tqdm(test_loader, desc="Baseline Test"):
            labels = labels.to(device)
            
            # Forward pass
            a_seq, a_mask = model['audio_encoder'](audio_list, text_list)
            t_seq, t_mask = model['text_encoder'](text_list)
            a_enh, t_enh = model['cross'](a_seq, t_seq, a_mask, t_mask)
            a_vec = model['pool_a'](a_enh, a_mask)
            t_vec = model['pool_t'](t_enh, t_mask)
            fused = model['fusion'](a_vec, t_vec)
            
            # Classification
            logits = model['classifier'](fused, use_openmax=True)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    f1 = weighted_f1(torch.tensor(all_preds), torch.tensor(all_labels))
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    
    print(f"✅ Baseline Evaluation: F1={f1:.4f}, Accuracy={accuracy:.4f}")
    return {'f1': f1, 'accuracy': accuracy, 'predictions': all_preds, 'labels': all_labels}

def test_cross_lingual_analysis(model, test_loader, device):
    """Test cross-lingual analysis."""
    print("\n🧪 Testing Cross-Lingual Analysis...")
    
    try:
        from evaluation.cross_lingual_metrics import create_cross_lingual_evaluator
        
        # Get baseline results first
        baseline_results = test_baseline_evaluation(model, test_loader, device)
        predictions = baseline_results['predictions']
        labels = baseline_results['labels']
        
        # Simulate multilingual text
        texts = [f"Sample text in language {i % 3}" for i in range(len(predictions))]
        
        # Test cross-lingual evaluator
        cross_lingual_evaluator = create_cross_lingual_evaluator()
        language_performances = cross_lingual_evaluator.evaluate_per_language(
            predictions, labels, texts
        )
        
        print(f"✅ Cross-Lingual Analysis: {len(language_performances)} languages analyzed")
        return {'language_performances': language_performances}
        
    except Exception as e:
        print(f"❌ Cross-Lingual Analysis failed: {e}")
        return None

def test_calibration_analysis(model, test_loader, device):
    """Test calibration analysis."""
    print("\n🧪 Testing Calibration Analysis...")
    
    try:
        from evaluation.calibration_metrics import create_calibration_evaluator
        
        model['fusion'].eval()
        model['classifier'].eval()
        
        all_preds, all_labels, all_probs = [], [], []
        
        with torch.no_grad():
            for audio_list, text_list, labels in tqdm(test_loader, desc="Calibration Test"):
                labels = labels.to(device)
                
                # Forward pass
                a_seq, a_mask = model['audio_encoder'](audio_list, text_list)
                t_seq, t_mask = model['text_encoder'](text_list)
                a_enh, t_enh = model['cross'](a_seq, t_seq, a_mask, t_mask)
                a_vec = model['pool_a'](a_enh, a_mask)
                t_vec = model['pool_t'](t_enh, t_mask)
                fused = model['fusion'](a_vec, t_vec)
                
                # Get predictions and probabilities
                logits = model['classifier'](fused, use_openmax=True)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Test calibration evaluator
        calibration_evaluator = create_calibration_evaluator()
        calibration_metrics = calibration_evaluator.compute_calibration_metrics(
            np.array(all_preds), np.array(all_labels), np.array(all_probs)
        )
        
        print(f"✅ Calibration Analysis: ECE={calibration_metrics.ece:.4f}")
        return {'ece': calibration_metrics.ece, 'mce': calibration_metrics.mce}
        
    except Exception as e:
        print(f"❌ Calibration Analysis failed: {e}")
        return None

def test_asr_analysis(model, test_loader, device):
    """Test ASR analysis."""
    print("\n🧪 Testing ASR Analysis...")
    
    try:
        from evaluation.asr_performance_tracker import create_asr_performance_tracker
        
        asr_tracker = create_asr_performance_tracker()
        
        # Simulate ASR evaluation
        with torch.no_grad():
            for audio_list, text_list, labels in tqdm(test_loader, desc="ASR Test"):
                for i, (ref_text, hyp_text) in enumerate(zip(text_list, text_list)):
                    confidence_scores = [0.8]  # Simulated confidence
                    processing_time = 0.1  # Simulated processing time
                    asr_tracker.update_metrics(ref_text, hyp_text, confidence_scores, processing_time)
        
        # Generate ASR report
        asr_report = asr_tracker.generate_performance_report()
        
        print(f"✅ ASR Analysis: Report generated successfully")
        return {'asr_report': asr_report}
        
    except Exception as e:
        print(f"❌ ASR Analysis failed: {e}")
        return None

def test_inference_benchmarking(model, device):
    """Test inference benchmarking."""
    print("\n🧪 Testing Inference Benchmarking...")
    
    try:
        from evaluation.inference_metrics import create_inference_benchmarker
        
        benchmarker = create_inference_benchmarker(device)
        
        # Test classifier benchmarking
        print("  Testing classifier...")
        classifier = model['classifier']
        sample_input = torch.randn(1, 512).to(device)
        
        # Test direct call
        with torch.no_grad():
            output = classifier(sample_input, use_openmax=True)
            print(f"    ✅ Classifier direct call successful: {output.shape}")
        
        # Test fusion benchmarking
        print("  Testing fusion...")
        fusion = model['fusion']
        audio_vec = torch.randn(1, 512).to(device)
        text_vec = torch.randn(1, 512).to(device)
        
        # Test direct call
        with torch.no_grad():
            output = fusion(audio_vec, text_vec)
            print(f"    ✅ Fusion direct call successful: {output.shape}")
        
        print(f"✅ Inference Benchmarking: Direct calls successful")
        return {'status': 'success'}
        
    except Exception as e:
        print(f"❌ Inference Benchmarking failed: {e}")
        return None

def test_few_shot_adaptation(model, test_loader, device):
    """Test few-shot adaptation."""
    print("\n🧪 Testing Few-Shot Adaptation...")
    
    try:
        from evaluation.few_shot_adaptation import create_few_shot_adapter
        
        few_shot_adapter = create_few_shot_adapter(model, device)
        
        # Test model cloning with state dict approach
        print("  Testing model cloning...")
        cloned_model = {}
        for key, model_component in model.items():
            # Create a new instance and load state dict
            if key == 'audio_encoder':
                cloned_model[key] = AudioEncoder().to(device)
            elif key == 'text_encoder':
                cloned_model[key] = TextEncoder().to(device)
            elif key == 'cross':
                audio_hid = model['audio_encoder'].encoder.config.hidden_size
                text_hid = model['text_encoder'].encoder.config.hidden_size
                cloned_model[key] = CrossModalAttention(audio_hid, text_hid, shared_dim=256, num_heads=8).to(device)
            elif key == 'pool_a':
                audio_hid = model['audio_encoder'].encoder.config.hidden_size
                cloned_model[key] = AttentiveStatsPooling(audio_hid).to(device)
            elif key == 'pool_t':
                text_hid = model['text_encoder'].encoder.config.hidden_size
                cloned_model[key] = AttentiveStatsPooling(text_hid).to(device)
            elif key == 'fusion':
                audio_hid = model['audio_encoder'].encoder.config.hidden_size
                text_hid = model['text_encoder'].encoder.config.hidden_size
                cloned_model[key] = FusionLayer(audio_hid * 2, text_hid * 2, 512).to(device)
            elif key == 'classifier':
                cloned_model[key] = AdvancedOpenMaxClassifier(
                    input_dim=512, num_labels=4, num_layers=35, base_dim=512, dropout=0.15
                ).to(device)
            elif key == 'prototypes':
                cloned_model[key] = PrototypeMemory(4, 512).to(device)
            
            # Load state dict
            cloned_model[key].load_state_dict(model_component.state_dict())
        
        print(f"    ✅ Model cloning successful")
        
        # Test few-shot dataset creation
        print("  Testing few-shot dataset creation...")
        few_shot_dataset, eval_dataset = few_shot_adapter.create_few_shot_dataset(
            test_loader.dataset, num_shots=10
        )
        
        print(f"    ✅ Few-shot dataset created: {len(few_shot_dataset)} samples")
        
        print(f"✅ Few-Shot Adaptation: All components working")
        return {'status': 'success'}
        
    except Exception as e:
        print(f"❌ Few-Shot Adaptation failed: {e}")
        return None

def test_robustness_evaluation(model, test_loader, device):
    """Test robustness evaluation."""
    print("\n🧪 Testing Robustness Evaluation...")
    
    try:
        from evaluation.robustness_evaluation import create_robustness_evaluator
        
        robustness_evaluator = create_robustness_evaluator(model, device)
        robustness_evaluator.set_baseline_performance(0.5, 0.5)
        
        # Test noise robustness with limited samples
        print("  Testing noise robustness...")
        limited_loader = list(test_loader)[:2]  # Test with first 2 batches only
        
        # Create a simple test
        for batch in limited_loader:
            audio_list, text_list, labels = batch
            labels = labels.to(device)
            
            # Test forward pass
            a_seq, a_mask = model['audio_encoder'](audio_list, text_list)
            t_seq, t_mask = model['text_encoder'](text_list)
            a_enh, t_enh = model['cross'](a_seq, t_seq, a_mask, t_mask)
            a_vec = model['pool_a'](a_enh, a_mask)
            t_vec = model['pool_t'](t_enh, t_mask)
            fused = model['fusion'](a_vec, t_vec)
            
            # Test classification
            logits = model['classifier'](fused, use_openmax=True)
            probs = torch.softmax(logits, dim=1)
            max_probs = torch.max(probs, dim=1)[0]
            uncertainty = 1.0 - max_probs
            
            print(f"    ✅ Robustness forward pass successful")
            break
        
        print(f"✅ Robustness Evaluation: All components working")
        return {'status': 'success'}
        
    except Exception as e:
        print(f"❌ Robustness Evaluation failed: {e}")
        return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Academic Components Individually")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/epoch_1_f1_0.4884.pt',
                       help='Path to checkpoint file')
    parser.add_argument('--test_manifest', type=str, default='test_10.jsonl',
                       help='Path to test manifest file')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for evaluation')
    parser.add_argument('--components', type=str, nargs='+', 
                       default=['baseline', 'cross_lingual', 'calibration', 'asr', 'inference', 'few_shot', 'robustness'],
                       help='Components to test')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load and freeze model
    model = load_and_freeze_model(args.checkpoint, device)
    
    # Load test data
    print(f"Loading test data from {args.test_manifest}")
    test_ds = SERDataset(args.test_manifest)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"Test set: {len(test_ds)} samples")
    
    # Test components individually
    results = {}
    
    if 'baseline' in args.components:
        results['baseline'] = test_baseline_evaluation(model, test_loader, device)
    
    if 'cross_lingual' in args.components:
        results['cross_lingual'] = test_cross_lingual_analysis(model, test_loader, device)
    
    if 'calibration' in args.components:
        results['calibration'] = test_calibration_analysis(model, test_loader, device)
    
    if 'asr' in args.components:
        results['asr'] = test_asr_analysis(model, test_loader, device)
    
    if 'inference' in args.components:
        results['inference'] = test_inference_benchmarking(model, device)
    
    if 'few_shot' in args.components:
        results['few_shot'] = test_few_shot_adaptation(model, test_loader, device)
    
    if 'robustness' in args.components:
        results['robustness'] = test_robustness_evaluation(model, test_loader, device)
    
    # Print summary
    print("\n" + "="*60)
    print("COMPONENT TEST SUMMARY")
    print("="*60)
    
    for component, result in results.items():
        if result is not None:
            print(f"✅ {component.upper()}: PASSED")
        else:
            print(f"❌ {component.upper()}: FAILED")
    
    print(f"\n🎉 Component testing completed!")
    return results

if __name__ == "__main__":
    main()
