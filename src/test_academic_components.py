#!/usr/bin/env python3
"""
Test script to verify all academic evaluation components work correctly.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test that all academic evaluation modules can be imported."""
    print("🧪 Testing Academic Component Imports...")
    
    try:
        from evaluation.cross_lingual_metrics import create_cross_lingual_evaluator
        print("✅ Cross-lingual metrics imported successfully")
    except ImportError as e:
        print(f"❌ Cross-lingual metrics import failed: {e}")
        return False
    
    try:
        from evaluation.calibration_metrics import create_calibration_evaluator
        print("✅ Calibration metrics imported successfully")
    except ImportError as e:
        print(f"❌ Calibration metrics import failed: {e}")
        return False
    
    try:
        from evaluation.asr_performance_tracker import create_asr_performance_tracker
        print("✅ ASR performance tracker imported successfully")
    except ImportError as e:
        print(f"❌ ASR performance tracker import failed: {e}")
        return False
    
    try:
        from evaluation.inference_metrics import create_inference_benchmarker
        print("✅ Inference metrics imported successfully")
    except ImportError as e:
        print(f"❌ Inference metrics import failed: {e}")
        return False
    
    try:
        from evaluation.few_shot_adaptation import create_few_shot_adapter
        print("✅ Few-shot adaptation imported successfully")
    except ImportError as e:
        print(f"❌ Few-shot adaptation import failed: {e}")
        return False
    
    try:
        from evaluation.robustness_evaluation import create_robustness_evaluator
        print("✅ Robustness evaluation imported successfully")
    except ImportError as e:
        print(f"❌ Robustness evaluation import failed: {e}")
        return False
    
    return True

def test_component_creation():
    """Test that all components can be created."""
    print("\n🧪 Testing Component Creation...")
    
    try:
        # Test cross-lingual evaluator
        cross_lingual_evaluator = create_cross_lingual_evaluator()
        print("✅ Cross-lingual evaluator created successfully")
    except Exception as e:
        print(f"❌ Cross-lingual evaluator creation failed: {e}")
        return False
    
    try:
        # Test calibration evaluator
        calibration_evaluator = create_calibration_evaluator()
        print("✅ Calibration evaluator created successfully")
    except Exception as e:
        print(f"❌ Calibration evaluator creation failed: {e}")
        return False
    
    try:
        # Test ASR performance tracker
        asr_tracker = create_asr_performance_tracker()
        print("✅ ASR performance tracker created successfully")
    except Exception as e:
        print(f"❌ ASR performance tracker creation failed: {e}")
        return False
    
    try:
        # Test inference benchmarker
        inference_benchmarker = create_inference_benchmarker()
        print("✅ Inference benchmarker created successfully")
    except Exception as e:
        print(f"❌ Inference benchmarker creation failed: {e}")
        return False
    
    try:
        # Test few-shot adapter (requires a mock model)
        class MockModel:
            def __init__(self):
                self.parameters = lambda: []
        
        mock_model = {'classifier': MockModel()}
        few_shot_adapter = create_few_shot_adapter(mock_model, 'cpu')
        print("✅ Few-shot adapter created successfully")
    except Exception as e:
        print(f"❌ Few-shot adapter creation failed: {e}")
        return False
    
    try:
        # Test robustness evaluator (requires a mock model)
        robustness_evaluator = create_robustness_evaluator(mock_model, 'cpu')
        print("✅ Robustness evaluator created successfully")
    except Exception as e:
        print(f"❌ Robustness evaluator creation failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of components."""
    print("\n🧪 Testing Basic Functionality...")
    
    try:
        # Test cross-lingual evaluator
        cross_lingual_evaluator = create_cross_lingual_evaluator()
        
        # Test with dummy data
        predictions = [0, 1, 2, 3, 0, 1, 2, 3]
        labels = [0, 1, 2, 3, 0, 1, 2, 3]
        texts = ["Sample text"] * 8
        
        language_performances = cross_lingual_evaluator.evaluate_per_language(
            predictions, labels, texts
        )
        print("✅ Cross-lingual evaluation completed")
        
    except Exception as e:
        print(f"❌ Cross-lingual evaluation failed: {e}")
        return False
    
    try:
        # Test calibration evaluator
        calibration_evaluator = create_calibration_evaluator()
        
        # Test with dummy data
        predictions = [0, 1, 2, 3, 0, 1, 2, 3]
        labels = [0, 1, 2, 3, 0, 1, 2, 3]
        probabilities = [[0.8, 0.1, 0.05, 0.05]] * 8
        
        calibration_metrics = calibration_evaluator.compute_calibration_metrics(
            predictions, labels, probabilities
        )
        print("✅ Calibration evaluation completed")
        
    except Exception as e:
        print(f"❌ Calibration evaluation failed: {e}")
        return False
    
    try:
        # Test ASR performance tracker
        asr_tracker = create_asr_performance_tracker()
        
        # Test with dummy data
        asr_tracker.update_metrics("reference text", "hypothesis text", [0.8], 0.1)
        print("✅ ASR performance tracking completed")
        
    except Exception as e:
        print(f"❌ ASR performance tracking failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🚀 Testing Academic Evaluation Components")
    print("=" * 50)
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ Import tests failed!")
        return False
    
    # Test 2: Component creation
    if not test_component_creation():
        print("\n❌ Component creation tests failed!")
        return False
    
    # Test 3: Basic functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality tests failed!")
        return False
    
    print("\n🎉 All academic component tests passed successfully!")
    print("✅ Your system is ready for comprehensive academic evaluation!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
