"""
Compatibility shim for tests expecting `src.models.asr_integration_fixed`.

Re-exports the factory `create_enhanced_asr` and the `ASRResult` dataclass
from `src.models.asr_integration`.
"""

from .asr_integration import (
    create_enhanced_asr,
    ASRResult,
    EnhancedASRIntegration,
)

__all__ = [
    "create_enhanced_asr",
    "ASRResult",
    "EnhancedASRIntegration",
]


