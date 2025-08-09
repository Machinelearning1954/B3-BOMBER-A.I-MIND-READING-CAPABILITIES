# B2 Spirit AI API Reference

## Overview

This document provides comprehensive API reference for the B2 Spirit AI system, including all modules, classes, and functions for Meta AI, Anduril, Palantir, and Neural Interface integrations.

## Table of Contents

1. [Core Fusion API](#core-fusion-api)
2. [Meta AI API](#meta-ai-api)
3. [Anduril Defense API](#anduril-defense-api)
4. [Palantir Analytics API](#palantir-analytics-api)
5. [Neural Interface API](#neural-interface-api)
6. [Utility Functions](#utility-functions)
7. [Configuration](#configuration)
8. [Error Handling](#error-handling)

## Core Fusion API

### EnhancedB2SpiritAI

Main system class that integrates all AI technologies with neural interface capabilities.

```python
class EnhancedB2SpiritAI:
    def __init__(self, config: EnhancedFusionConfig)
    async def enhanced_analyze(self, **kwargs) -> EnhancedAnalysisResult
    def generate_strategy(self, **kwargs) -> Dict[str, Any]
```

#### Constructor

```python
EnhancedB2SpiritAI(config: EnhancedFusionConfig)
```

**Parameters:**
- `config` (EnhancedFusionConfig): System configuration object

**Example:**
```python
from src.fusion import EnhancedB2SpiritAI, EnhancedFusionConfig

config = EnhancedFusionConfig(
    neural_interface_enabled=True,
    mind_reading_threshold=0.85,
    language_interpretation_enabled=True
)
system = EnhancedB2SpiritAI(config)
```

#### enhanced_analyze()

Performs comprehensive multi-modal analysis with neural interface integration.

```python
async def enhanced_analyze(
    self,
    image_data: Optional[np.ndarray] = None,
    text_data: Optional[str] = None,
    sensor_data: Optional[Dict[str, Any]] = None,
    neural_signal: Optional[NeuralSignal] = None,
    mission_context: Optional[Dict[str, Any]] = None
) -> EnhancedAnalysisResult
```

**Parameters:**
- `image_data` (np.ndarray, optional): Satellite imagery, reconnaissance photos
- `text_data` (str, optional): Mission briefings, intelligence reports
- `sensor_data` (Dict[str, Any], optional): Radar, LIDAR, thermal sensor readings
- `neural_signal` (NeuralSignal, optional): Neural interface data from operator
- `mission_context` (Dict[str, Any], optional): Mission parameters and constraints

**Returns:**
- `EnhancedAnalysisResult`: Comprehensive analysis with neural insights

**Example:**
```python
result = await system.enhanced_analyze(
    image_data=satellite_image,
    sensor_data={'radar': radar_data, 'thermal': thermal_data},
    neural_signal=operator_neural_signal,
    mission_context={'phase': 'reconnaissance', 'priority': 'high'}
)

print(f"Threats detected: {len(result.threats)}")
print(f"Neural commands: {result.neural_commands}")
print(f"Mind-machine sync: {result.mind_machine_sync:.2f}")
```

### EnhancedFusionConfig

Configuration class for the enhanced B2 Spirit AI system.

```python
@dataclass
class EnhancedFusionConfig:
    meta_weight: float = 0.25
    anduril_weight: float = 0.25
    palantir_weight: float = 0.25
    neural_interface_weight: float = 0.25
    fusion_method: str = "neural_attention_weighted"
    confidence_threshold: float = 0.90
    max_parallel_tasks: int = 12
    enable_quantum_enhancement: bool = True
    use_federated_learning: bool = True
    neural_interface_enabled: bool = True
    mind_reading_threshold: float = 0.85
    language_interpretation_enabled: bool = True
    cognitive_monitoring: bool = True
```

### EnhancedAnalysisResult

Result object containing comprehensive analysis with neural interface data.

```python
@dataclass
class EnhancedAnalysisResult:
    threats: List[Dict[str, Any]]
    opportunities: List[Dict[str, Any]]
    recommendations: List[str]
    confidence_score: float
    processing_time: float
    metadata: Dict[str, Any]
    decoded_thoughts: Optional[ThoughtDecoding] = None
    operator_cognitive_state: Optional[Dict[str, Any]] = None
    neural_commands: Optional[List[str]] = None
    mind_machine_sync: Optional[float] = None
```

## Neural Interface API

### AdvancedMindReader

Main neural interface system for mind reading and thought decoding.

```python
class AdvancedMindReader:
    def __init__(self, model_path: Optional[str] = None)
    async def decode_thoughts(self, neural_signal: NeuralSignal) -> ThoughtDecoding
    def load_model(self, model_path: str)
    def save_model(self, model_path: str)
```

#### decode_thoughts()

Decodes thoughts from neural signals with language interpretation.

```python
async def decode_thoughts(self, neural_signal: NeuralSignal) -> ThoughtDecoding
```

**Parameters:**
- `neural_signal` (NeuralSignal): Raw neural signal data

**Returns:**
- `ThoughtDecoding`: Decoded thought information with language interpretation

**Example:**
```python
from src.neural_interface import AdvancedMindReader, NeuralSignal

mind_reader = AdvancedMindReader()

neural_signal = NeuralSignal(
    eeg_data=eeg_array,
    sampling_rate=1000.0,
    channels=['Fp1', 'Fp2', 'F3', 'F4'],
    subject_id="operator_001"
)

thoughts = await mind_reader.decode_thoughts(neural_signal)
print(f"Decoded: {thoughts.text_content}")
print(f"Language: {thoughts.language}")
print(f"Intent: {thoughts.intent}")
```

### NeuralSignal

Data structure for neural signal information.

```python
@dataclass
class NeuralSignal:
    eeg_data: np.ndarray
    fmri_data: Optional[np.ndarray] = None
    sampling_rate: float = 1000.0
    channels: List[str] = None
    timestamp: float = 0.0
    subject_id: str = "unknown"
    session_id: str = "default"
```

**Attributes:**
- `eeg_data` (np.ndarray): EEG signal data, shape (channels, time_points)
- `fmri_data` (np.ndarray, optional): fMRI data for enhanced accuracy
- `sampling_rate` (float): Signal sampling rate in Hz
- `channels` (List[str]): Channel names/labels
- `timestamp` (float): Signal timestamp
- `subject_id` (str): Operator/subject identifier
- `session_id` (str): Session identifier

### ThoughtDecoding

Result of neural thought decoding with comprehensive analysis.

```python
@dataclass
class ThoughtDecoding:
    text_content: str
    confidence_score: float
    language: str
    emotion: str
    intent: str
    cognitive_load: float
    attention_level: float
    stress_indicators: Dict[str, float]
    neural_patterns: Dict[str, Any]
```

**Attributes:**
- `text_content` (str): Decoded text from neural signals
- `confidence_score` (float): Overall decoding confidence (0.0-1.0)
- `language` (str): Detected language code
- `emotion` (str): Detected emotional state
- `intent` (str): Classified intent/purpose
- `cognitive_load` (float): Cognitive load level (0.0-1.0)
- `attention_level` (float): Attention level (0.0-1.0)
- `stress_indicators` (Dict): Stress-related measurements
- `neural_patterns` (Dict): Raw neural pattern data

### NeuralLanguageInterpreter

Advanced multilingual neural language interpretation system.

```python
class NeuralLanguageInterpreter:
    def __init__(self, model_path: Optional[str] = None)
    async def interpret_neural_language(self, **kwargs) -> InterpretedLanguage
    def load_model(self, model_path: str)
    def save_model(self, model_path: str)
```

#### interpret_neural_language()

Interprets neural language signals with contextual processing.

```python
async def interpret_neural_language(
    self,
    neural_features: torch.Tensor,
    raw_text: str,
    context: LanguageContext
) -> InterpretedLanguage
```

**Parameters:**
- `neural_features` (torch.Tensor): Processed neural signal features
- `raw_text` (str): Raw decoded text from neural signals
- `context` (LanguageContext): Language and contextual information

**Returns:**
- `InterpretedLanguage`: Comprehensive language interpretation result

### LanguageContext

Context information for neural language interpretation.

```python
@dataclass
class LanguageContext:
    primary_language: str = "en"
    secondary_languages: List[str] = None
    domain_context: str = "military"
    urgency_level: str = "normal"
    cognitive_state: str = "normal"
    mission_phase: str = "operational"
```

### InterpretedLanguage

Result of neural language interpretation with actionable insights.

```python
@dataclass
class InterpretedLanguage:
    original_text: str
    interpreted_text: str
    language_detected: str
    confidence_score: float
    intent_classification: str
    urgency_level: str
    actionable_commands: List[str]
    emotional_context: str
    cognitive_markers: Dict[str, float]
    translation_quality: float
    context_awareness: float
```

### NeuralSignalProcessor

Advanced neural signal processing for EEG and fMRI data.

```python
class NeuralSignalProcessor:
    def __init__(self, sampling_rate: float = 1000.0)
    def preprocess_eeg(self, raw_eeg: np.ndarray) -> np.ndarray
    def extract_frequency_features(self, eeg_data: np.ndarray) -> Dict[str, np.ndarray]
```

#### preprocess_eeg()

Preprocesses raw EEG signals with filtering and artifact removal.

```python
def preprocess_eeg(self, raw_eeg: np.ndarray) -> np.ndarray
```

**Parameters:**
- `raw_eeg` (np.ndarray): Raw EEG data, shape (channels, time)

**Returns:**
- `np.ndarray`: Preprocessed EEG signals

**Example:**
```python
processor = NeuralSignalProcessor(sampling_rate=1000.0)
cleaned_eeg = processor.preprocess_eeg(raw_eeg_data)
features = processor.extract_frequency_features(cleaned_eeg)
```

## Meta AI API

### MetaAISystem

Integration system for Meta AI models and capabilities.

```python
class MetaAISystem:
    def __init__(self)
    async def segment_anything(self, image: np.ndarray) -> Dict[str, Any]
    async def describe_scene(self, image: np.ndarray) -> str
    async def detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]
    async def understand_text(self, text: str) -> Dict[str, Any]
    async def extract_entities(self, text: str) -> List[Dict[str, Any]]
    async def generate_embeddings(self, text: str) -> np.ndarray
    def validate_strategy(self, strategy: Dict, enhancements: Dict) -> Dict[str, Any]
```

#### segment_anything()

Performs image segmentation using Meta's SAM model.

```python
async def segment_anything(self, image: np.ndarray) -> Dict[str, Any]
```

**Parameters:**
- `image` (np.ndarray): Input image array

**Returns:**
- `Dict[str, Any]`: Segmentation results with masks and metadata

#### understand_text()

Processes text using LLaMA language models for understanding.

```python
async def understand_text(self, text: str) -> Dict[str, Any]
```

**Parameters:**
- `text` (str): Input text to analyze

**Returns:**
- `Dict[str, Any]`: Text understanding results including sentiment, topics, and insights

## Anduril Defense API

### AndurilDefenseSystem

Integration system for Anduril defense technologies.

```python
class AndurilDefenseSystem:
    def __init__(self)
    async def process_sensor_data(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]
    async def assess_threats(self, sensor_data: Dict[str, Any]) -> List[Dict[str, Any]]
    async def generate_recommendations(self, sensor_data: Dict[str, Any]) -> List[str]
    def enhance_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]
```

#### assess_threats()

Analyzes sensor data for threat detection and assessment.

```python
async def assess_threats(self, sensor_data: Dict[str, Any]) -> List[Dict[str, Any]]
```

**Parameters:**
- `sensor_data` (Dict[str, Any]): Multi-modal sensor readings

**Returns:**
- `List[Dict[str, Any]]`: Detected threats with classification and confidence

**Example:**
```python
anduril = AndurilDefenseSystem()
threats = await anduril.assess_threats({
    'radar': radar_readings,
    'thermal': thermal_data,
    'acoustic': audio_signatures
})
```

## Palantir Analytics API

### PalantirAnalytics

Integration system for Palantir data fusion and analytics.

```python
class PalantirAnalytics:
    def __init__(self)
    def generate_strategy(self, **kwargs) -> Dict[str, Any]
    def analyze_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]
    def fuse_intelligence(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]
```

#### generate_strategy()

Generates strategic recommendations based on comprehensive analysis.

```python
def generate_strategy(
    self,
    threats: List[Dict[str, Any]],
    objectives: List[str],
    constraints: Dict[str, Any]
) -> Dict[str, Any]
```

**Parameters:**
- `threats` (List[Dict]): Identified threats and their characteristics
- `objectives` (List[str]): Mission objectives and goals
- `constraints` (Dict[str, Any]): Resource and operational constraints

**Returns:**
- `Dict[str, Any]`: Strategic plan with prioritized actions

## Utility Functions

### Configuration Management

```python
def load_config(config_path: str) -> Dict[str, Any]
def save_config(config: Dict[str, Any], config_path: str)
def validate_config(config: Dict[str, Any]) -> bool
```

### Logging and Monitoring

```python
def setup_logging(level: str = "INFO", log_file: Optional[str] = None)
def performance_monitor(func: Callable) -> Callable
def log_neural_activity(neural_signal: NeuralSignal, thoughts: ThoughtDecoding)
```

### Input Validation

```python
def validate_inputs(func: Callable) -> Callable
def validate_neural_signal(signal: NeuralSignal) -> bool
def validate_image_data(image: np.ndarray) -> bool
def validate_sensor_data(data: Dict[str, Any]) -> bool
```

### Data Processing

```python
def normalize_neural_data(data: np.ndarray) -> np.ndarray
def extract_neural_features(signal: NeuralSignal) -> torch.Tensor
def preprocess_image(image: np.ndarray) -> np.ndarray
def format_sensor_data(raw_data: Dict[str, Any]) -> Dict[str, Any]
```

## Configuration

### System Configuration Files

#### Main Configuration (`configs/main.yaml`)

```yaml
system:
  name: "B2 Spirit AI"
  version: "1.0.0"
  environment: "production"

fusion:
  meta_weight: 0.25
  anduril_weight: 0.25
  palantir_weight: 0.25
  neural_interface_weight: 0.25
  fusion_method: "neural_attention_weighted"
  confidence_threshold: 0.90

neural_interface:
  enabled: true
  mind_reading_threshold: 0.85
  language_interpretation: true
  cognitive_monitoring: true
  supported_languages: ["en", "es", "fr", "de", "zh", "ja", "ar", "ru"]

performance:
  max_parallel_tasks: 12
  gpu_acceleration: true
  memory_optimization: true
  real_time_processing: true
```

#### Neural Interface Configuration (`configs/neural_interface.yaml`)

```yaml
signal_acquisition:
  eeg:
    channels: 64
    sampling_rate: 1000
    frequency_range: [0.5, 50]
  fmri:
    enabled: false
    temporal_resolution: 2.0
    spatial_resolution: 2.0

processing:
  preprocessing:
    artifact_removal: true
    frequency_filtering: true
    spatial_filtering: true
  feature_extraction:
    frequency_bands: true
    time_domain: true
    spatial_patterns: true

models:
  language_decoder:
    model_type: "transformer"
    hidden_dim: 512
    num_layers: 6
    vocab_size: 50000
  cognitive_analyzer:
    emotion_classes: 7
    stress_levels: 3
    attention_tracking: true
```

### Environment Variables

```bash
# Core system
export B2_SPIRIT_CONFIG_PATH="/path/to/configs"
export B2_SPIRIT_MODEL_PATH="/path/to/models"
export B2_SPIRIT_LOG_LEVEL="INFO"

# Neural interface
export NEURAL_INTERFACE_ENABLED="true"
export MIND_READING_THRESHOLD="0.85"
export LANGUAGE_INTERPRETATION="true"

# Performance
export GPU_ACCELERATION="true"
export MAX_PARALLEL_TASKS="12"
export REAL_TIME_PROCESSING="true"

# Security
export NEURAL_DATA_ENCRYPTION="true"
export ACCESS_CONTROL_ENABLED="true"
export AUDIT_LOGGING="true"
```

## Error Handling

### Exception Classes

```python
class B2SpiritAIError(Exception):
    """Base exception for B2 Spirit AI system"""
    pass

class NeuralInterfaceError(B2SpiritAIError):
    """Neural interface specific errors"""
    pass

class SignalProcessingError(NeuralInterfaceError):
    """Signal processing errors"""
    pass

class LanguageDecodingError(NeuralInterfaceError):
    """Language decoding errors"""
    pass

class CognitiveAnalysisError(NeuralInterfaceError):
    """Cognitive analysis errors"""
    pass

class FusionError(B2SpiritAIError):
    """System fusion errors"""
    pass

class ConfigurationError(B2SpiritAIError):
    """Configuration errors"""
    pass
```

### Error Handling Patterns

```python
try:
    # Neural interface operations
    thoughts = await mind_reader.decode_thoughts(neural_signal)
except SignalProcessingError as e:
    logger.error(f"Signal processing failed: {e}")
    # Fallback to manual input
    thoughts = get_manual_input()
except LanguageDecodingError as e:
    logger.error(f"Language decoding failed: {e}")
    # Use simplified decoding
    thoughts = simple_decode(neural_signal)
except NeuralInterfaceError as e:
    logger.error(f"Neural interface error: {e}")
    # Disable neural interface temporarily
    disable_neural_interface()
```

### Logging and Debugging

```python
import logging
from src.utils import setup_logging

# Setup comprehensive logging
setup_logging(level="DEBUG", log_file="b2_spirit.log")

# Neural interface specific logging
neural_logger = logging.getLogger("neural_interface")
neural_logger.info("Neural interface initialized")

# Performance monitoring
@performance_monitor
async def monitored_analysis():
    result = await system.enhanced_analyze(...)
    return result
```

### Health Checks

```python
def system_health_check() -> Dict[str, bool]:
    """Comprehensive system health check"""
    return {
        'neural_interface': check_neural_interface_health(),
        'signal_quality': check_signal_quality(),
        'model_status': check_model_status(),
        'fusion_system': check_fusion_system(),
        'memory_usage': check_memory_usage(),
        'gpu_status': check_gpu_status()
    }

def check_neural_interface_health() -> bool:
    """Check neural interface system health"""
    try:
        # Test signal acquisition
        test_signal = generate_test_signal()
        processed = processor.preprocess_eeg(test_signal)
        return processed is not None
    except Exception:
        return False
```

---

For additional examples and advanced usage patterns, please refer to the [User Guide](user_guide.md) and [Neural Interface Guide](neural_interface_guide.md).

