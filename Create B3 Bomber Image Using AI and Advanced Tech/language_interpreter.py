"""
Neural Language Interpretation System

Advanced language interpretation module that processes decoded neural signals
and translates them into actionable commands and meaningful communication
across multiple languages and cognitive contexts.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import asyncio
import logging
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM,
    pipeline, MarianMTModel, MarianTokenizer
)
import spacy
from collections import defaultdict
import re

logger = logging.getLogger(__name__)


@dataclass
class LanguageContext:
    """Context information for language interpretation."""
    primary_language: str = "en"
    secondary_languages: List[str] = None
    domain_context: str = "military"
    urgency_level: str = "normal"
    cognitive_state: str = "normal"
    mission_phase: str = "operational"


@dataclass
class InterpretedLanguage:
    """Result of neural language interpretation."""
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


class MultilingualNeuralDecoder(nn.Module):
    """
    Advanced multilingual neural decoder that processes neural signals
    and converts them to language across multiple languages and contexts.
    """
    
    def __init__(self, 
                 input_dim: int = 1024,
                 vocab_size: int = 50000,
                 num_languages: int = 20,
                 hidden_dim: int = 768):
        super().__init__()
        
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.num_languages = num_languages
        self.hidden_dim = hidden_dim
        
        # Language-specific encoders
        self.language_encoders = nn.ModuleDict({
            'en': self._build_language_encoder(),
            'es': self._build_language_encoder(),
            'fr': self._build_language_encoder(),
            'de': self._build_language_encoder(),
            'zh': self._build_language_encoder(),
            'ja': self._build_language_encoder(),
            'ar': self._build_language_encoder(),
            'ru': self._build_language_encoder(),
        })
        
        # Cross-lingual attention mechanism
        self.cross_lingual_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=12, batch_first=True
        )
        
        # Language detection network
        self.language_detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_languages),
            nn.Softmax(dim=-1)
        )
        
        # Intent classification network
        self.intent_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 16)  # 16 intent categories
        )
        
        # Urgency detection network
        self.urgency_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 4)  # Low, Normal, High, Critical
        )
        
        # Cognitive state analyzer
        self.cognitive_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 8)  # Various cognitive markers
        )
        
    def _build_language_encoder(self) -> nn.Module:
        """Build language-specific encoder."""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
    
    def forward(self, neural_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process neural features and generate multilingual interpretations.
        
        Args:
            neural_features: Processed neural signal features
            
        Returns:
            Dictionary containing language predictions and analysis
        """
        batch_size = neural_features.size(0)
        
        # Detect primary language
        language_probs = self.language_detector(neural_features)
        
        # Encode in multiple languages
        language_encodings = {}
        for lang_code, encoder in self.language_encoders.items():
            language_encodings[lang_code] = encoder(neural_features)
        
        # Stack all language encodings for cross-lingual attention
        all_encodings = torch.stack(list(language_encodings.values()), dim=1)
        
        # Apply cross-lingual attention
        attended_encodings, attention_weights = self.cross_lingual_attention(
            all_encodings, all_encodings, all_encodings
        )
        
        # Get primary language encoding (highest probability)
        primary_lang_idx = torch.argmax(language_probs, dim=-1)
        primary_encoding = attended_encodings[torch.arange(batch_size), primary_lang_idx]
        
        # Classify intent
        intent_logits = self.intent_classifier(primary_encoding)
        
        # Detect urgency
        urgency_logits = self.urgency_detector(primary_encoding)
        
        # Analyze cognitive state
        cognitive_logits = self.cognitive_analyzer(primary_encoding)
        
        return {
            'language_probs': language_probs,
            'language_encodings': language_encodings,
            'attention_weights': attention_weights,
            'intent_logits': intent_logits,
            'urgency_logits': urgency_logits,
            'cognitive_logits': cognitive_logits,
            'primary_encoding': primary_encoding
        }


class ContextualTranslator:
    """
    Advanced contextual translator that handles military terminology,
    technical jargon, and domain-specific language patterns.
    """
    
    def __init__(self):
        self.translation_models = {}
        self.tokenizers = {}
        self.load_translation_models()
        
        # Military and technical terminology dictionaries
        self.military_terms = self._load_military_terminology()
        self.technical_terms = self._load_technical_terminology()
        
        # Context-aware translation patterns
        self.context_patterns = self._load_context_patterns()
        
    def load_translation_models(self):
        """Load pre-trained translation models for major language pairs."""
        language_pairs = [
            ('en', 'es'), ('en', 'fr'), ('en', 'de'), ('en', 'zh'),
            ('en', 'ja'), ('en', 'ar'), ('en', 'ru')
        ]
        
        for src, tgt in language_pairs:
            model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
            try:
                self.translation_models[f"{src}-{tgt}"] = MarianMTModel.from_pretrained(model_name)
                self.tokenizers[f"{src}-{tgt}"] = MarianTokenizer.from_pretrained(model_name)
            except Exception as e:
                logger.warning(f"Could not load translation model {model_name}: {e}")
    
    def _load_military_terminology(self) -> Dict[str, Dict[str, str]]:
        """Load military terminology translations."""
        return {
            'en': {
                'threat': 'threat',
                'hostile': 'hostile',
                'friendly': 'friendly',
                'target': 'target',
                'mission': 'mission',
                'sector': 'sector',
                'coordinates': 'coordinates',
                'engagement': 'engagement',
                'stealth': 'stealth',
                'reconnaissance': 'reconnaissance'
            },
            'es': {
                'threat': 'amenaza',
                'hostile': 'hostil',
                'friendly': 'amigable',
                'target': 'objetivo',
                'mission': 'misión',
                'sector': 'sector',
                'coordinates': 'coordenadas',
                'engagement': 'enfrentamiento',
                'stealth': 'sigilo',
                'reconnaissance': 'reconocimiento'
            },
            # Add more languages as needed
        }
    
    def _load_technical_terminology(self) -> Dict[str, Dict[str, str]]:
        """Load technical terminology translations."""
        return {
            'en': {
                'radar': 'radar',
                'sensor': 'sensor',
                'algorithm': 'algorithm',
                'neural': 'neural',
                'interface': 'interface',
                'processor': 'processor',
                'frequency': 'frequency',
                'bandwidth': 'bandwidth',
                'encryption': 'encryption',
                'protocol': 'protocol'
            },
            # Add translations for other languages
        }
    
    def _load_context_patterns(self) -> Dict[str, List[str]]:
        """Load context-aware translation patterns."""
        return {
            'military_command': [
                r'\b(deploy|engage|retreat|advance|hold)\b',
                r'\b(alpha|bravo|charlie|delta|echo)\b',
                r'\b(sector|grid|zone|area)\s+\w+\b'
            ],
            'technical_instruction': [
                r'\b(initialize|configure|calibrate|optimize)\b',
                r'\b(system|module|component|interface)\b',
                r'\b(frequency|bandwidth|signal|data)\b'
            ],
            'emergency': [
                r'\b(emergency|urgent|critical|immediate)\b',
                r'\b(help|assist|support|backup)\b',
                r'\b(mayday|sos|distress)\b'
            ]
        }
    
    async def translate_with_context(self, 
                                   text: str,
                                   source_lang: str,
                                   target_lang: str,
                                   context: LanguageContext) -> Tuple[str, float]:
        """
        Translate text with contextual awareness.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            context: Language context information
            
        Returns:
            Translated text and quality score
        """
        # Preprocess text with domain-specific terminology
        preprocessed_text = self._preprocess_domain_text(text, context)
        
        # Perform base translation
        model_key = f"{source_lang}-{target_lang}"
        if model_key in self.translation_models:
            translated_text = await self._neural_translate(
                preprocessed_text, model_key
            )
        else:
            # Fallback to rule-based translation
            translated_text = self._rule_based_translate(
                preprocessed_text, source_lang, target_lang
            )
        
        # Post-process with context-aware corrections
        final_text = self._postprocess_context_text(
            translated_text, target_lang, context
        )
        
        # Calculate translation quality
        quality_score = self._assess_translation_quality(
            text, final_text, source_lang, target_lang
        )
        
        return final_text, quality_score
    
    def _preprocess_domain_text(self, text: str, context: LanguageContext) -> str:
        """Preprocess text with domain-specific handling."""
        processed_text = text
        
        # Handle military terminology
        if context.domain_context == "military":
            for pattern_type, patterns in self.context_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, processed_text, re.IGNORECASE)
                    for match in matches:
                        # Mark military terms for special handling
                        processed_text = processed_text.replace(
                            match, f"<MILITARY:{match}>"
                        )
        
        return processed_text
    
    async def _neural_translate(self, text: str, model_key: str) -> str:
        """Perform neural machine translation."""
        try:
            model = self.translation_models[model_key]
            tokenizer = self.tokenizers[model_key]
            
            # Tokenize input
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            
            # Generate translation
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=512, num_beams=4)
            
            # Decode output
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return translated_text
            
        except Exception as e:
            logger.error(f"Neural translation failed: {e}")
            return text  # Return original text as fallback
    
    def _rule_based_translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Fallback rule-based translation."""
        # Simple word-by-word translation using terminology dictionaries
        words = text.split()
        translated_words = []
        
        source_terms = self.military_terms.get(source_lang, {})
        target_terms = self.military_terms.get(target_lang, {})
        
        # Create reverse mapping
        term_mapping = {}
        for en_term, source_term in source_terms.items():
            if en_term in target_terms:
                term_mapping[source_term.lower()] = target_terms[en_term]
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word in term_mapping:
                translated_words.append(term_mapping[clean_word])
            else:
                translated_words.append(word)  # Keep original if no translation
        
        return ' '.join(translated_words)
    
    def _postprocess_context_text(self, text: str, target_lang: str, context: LanguageContext) -> str:
        """Post-process translated text with context awareness."""
        processed_text = text
        
        # Remove military term markers and apply proper translations
        military_markers = re.findall(r'<MILITARY:([^>]+)>', processed_text)
        for marker in military_markers:
            # Apply context-specific translation
            if context.domain_context == "military":
                proper_translation = self._get_military_translation(marker, target_lang)
                processed_text = processed_text.replace(
                    f"<MILITARY:{marker}>", proper_translation
                )
        
        # Apply urgency-specific formatting
        if context.urgency_level in ["high", "critical"]:
            processed_text = processed_text.upper()
        
        return processed_text
    
    def _get_military_translation(self, term: str, target_lang: str) -> str:
        """Get proper military term translation."""
        # Find English equivalent first
        en_term = None
        for lang_terms in self.military_terms.values():
            for en, foreign in lang_terms.items():
                if foreign.lower() == term.lower():
                    en_term = en
                    break
            if en_term:
                break
        
        # Get target language translation
        if en_term and target_lang in self.military_terms:
            return self.military_terms[target_lang].get(en_term, term)
        
        return term
    
    def _assess_translation_quality(self, source: str, target: str, source_lang: str, target_lang: str) -> float:
        """Assess translation quality using various metrics."""
        # Simple quality assessment based on length ratio and term preservation
        length_ratio = len(target) / max(len(source), 1)
        
        # Penalize extreme length differences
        length_score = 1.0 - abs(1.0 - length_ratio) if 0.5 <= length_ratio <= 2.0 else 0.5
        
        # Check preservation of important terms
        important_terms = re.findall(r'\b[A-Z]{2,}\b', source)  # Acronyms and codes
        preserved_terms = sum(1 for term in important_terms if term in target)
        term_preservation = preserved_terms / max(len(important_terms), 1)
        
        # Overall quality score
        quality_score = 0.6 * length_score + 0.4 * term_preservation
        
        return min(max(quality_score, 0.0), 1.0)


class NeuralLanguageInterpreter:
    """
    Main neural language interpretation system that combines neural decoding,
    multilingual processing, and contextual translation.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the neural language interpreter.
        
        Args:
            model_path: Path to pre-trained model weights
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.neural_decoder = MultilingualNeuralDecoder()
        self.translator = ContextualTranslator()
        
        # Load language processing models
        self.nlp_models = {}
        self._load_nlp_models()
        
        # Intent and command mappings
        self.intent_labels = [
            'command', 'question', 'report', 'request', 'warning',
            'confirmation', 'denial', 'emergency', 'routine',
            'tactical', 'strategic', 'technical', 'medical',
            'communication', 'navigation', 'unknown'
        ]
        
        self.urgency_labels = ['low', 'normal', 'high', 'critical']
        
        self.cognitive_markers = [
            'attention', 'confusion', 'certainty', 'stress',
            'fatigue', 'focus', 'comprehension', 'decision_confidence'
        ]
        
        # Load pre-trained weights if available
        if model_path:
            self.load_model(model_path)
        
        self.logger.info("Neural Language Interpreter initialized successfully")
    
    def _load_nlp_models(self):
        """Load NLP models for various languages."""
        languages = ['en', 'es', 'fr', 'de']
        
        for lang in languages:
            try:
                model_name = f"{lang}_core_web_sm"
                self.nlp_models[lang] = spacy.load(model_name)
            except OSError:
                self.logger.warning(f"SpaCy model for {lang} not found. Using English model as fallback.")
                if 'en' not in self.nlp_models:
                    try:
                        self.nlp_models['en'] = spacy.load("en_core_web_sm")
                    except OSError:
                        self.logger.error("English SpaCy model not found. Some features may be limited.")
    
    async def interpret_neural_language(self,
                                      neural_features: torch.Tensor,
                                      raw_text: str,
                                      context: LanguageContext) -> InterpretedLanguage:
        """
        Interpret neural language signals with full contextual processing.
        
        Args:
            neural_features: Processed neural signal features
            raw_text: Raw decoded text from neural signals
            context: Language and contextual information
            
        Returns:
            Comprehensive language interpretation result
        """
        try:
            # Neural decoding and analysis
            neural_output = self.neural_decoder(neural_features)
            
            # Extract language probabilities and detect primary language
            language_probs = neural_output['language_probs']
            detected_lang_idx = torch.argmax(language_probs, dim=-1).item()
            language_codes = list(self.neural_decoder.language_encoders.keys())
            detected_language = language_codes[detected_lang_idx] if detected_lang_idx < len(language_codes) else 'en'
            
            # Classify intent
            intent_logits = neural_output['intent_logits']
            intent_idx = torch.argmax(intent_logits, dim=-1).item()
            intent_classification = self.intent_labels[intent_idx] if intent_idx < len(self.intent_labels) else 'unknown'
            
            # Detect urgency
            urgency_logits = neural_output['urgency_logits']
            urgency_idx = torch.argmax(urgency_logits, dim=-1).item()
            urgency_level = self.urgency_labels[urgency_idx] if urgency_idx < len(self.urgency_labels) else 'normal'
            
            # Analyze cognitive markers
            cognitive_logits = neural_output['cognitive_logits']
            cognitive_scores = torch.softmax(cognitive_logits, dim=-1)
            cognitive_markers = {
                marker: score.item() 
                for marker, score in zip(self.cognitive_markers, cognitive_scores[0])
            }
            
            # Translate and interpret text if needed
            if detected_language != context.primary_language:
                interpreted_text, translation_quality = await self.translator.translate_with_context(
                    raw_text, detected_language, context.primary_language, context
                )
            else:
                interpreted_text = raw_text
                translation_quality = 1.0
            
            # Extract actionable commands
            actionable_commands = await self._extract_actionable_commands(
                interpreted_text, intent_classification, context
            )
            
            # Analyze emotional context
            emotional_context = self._analyze_emotional_context(
                interpreted_text, cognitive_markers
            )
            
            # Calculate overall confidence
            confidence_score = self._calculate_interpretation_confidence(
                neural_output, translation_quality, cognitive_markers
            )
            
            # Assess context awareness
            context_awareness = self._assess_context_awareness(
                interpreted_text, context, cognitive_markers
            )
            
            return InterpretedLanguage(
                original_text=raw_text,
                interpreted_text=interpreted_text,
                language_detected=detected_language,
                confidence_score=confidence_score,
                intent_classification=intent_classification,
                urgency_level=urgency_level,
                actionable_commands=actionable_commands,
                emotional_context=emotional_context,
                cognitive_markers=cognitive_markers,
                translation_quality=translation_quality,
                context_awareness=context_awareness
            )
            
        except Exception as e:
            self.logger.error(f"Neural language interpretation failed: {str(e)}")
            raise
    
    async def _extract_actionable_commands(self,
                                         text: str,
                                         intent: str,
                                         context: LanguageContext) -> List[str]:
        """Extract actionable commands from interpreted text."""
        commands = []
        text_lower = text.lower()
        
        # Command patterns based on intent and context
        if intent == 'command':
            command_patterns = {
                'movement': [r'\b(move|go|navigate|proceed)\s+to\s+(\w+)', 
                           r'\b(advance|retreat|hold|position)\b'],
                'engagement': [r'\b(engage|fire|attack|target)\s+(\w+)',
                             r'\b(deploy|activate|launch)\s+(\w+)'],
                'communication': [r'\b(contact|signal|transmit|broadcast)\s+(\w+)',
                                r'\b(report|inform|notify)\s+(\w+)'],
                'reconnaissance': [r'\b(scan|search|monitor|observe)\s+(\w+)',
                                 r'\b(investigate|examine|survey)\s+(\w+)'],
                'defensive': [r'\b(shield|protect|defend|cover)\s+(\w+)',
                            r'\b(evade|avoid|escape)\s+(\w+)']
            }
            
            for category, patterns in command_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, text_lower)
                    for match in matches:
                        if isinstance(match, tuple):
                            command = f"{category}:{match[0]}:{match[1] if len(match) > 1 else ''}"
                        else:
                            command = f"{category}:{match}"
                        commands.append(command)
        
        elif intent == 'emergency':
            emergency_patterns = [
                r'\b(help|assist|support|backup)\b',
                r'\b(emergency|urgent|critical|mayday)\b',
                r'\b(medical|wounded|injured)\b'
            ]
            
            for pattern in emergency_patterns:
                if re.search(pattern, text_lower):
                    commands.append(f"emergency:{pattern}")
        
        return commands
    
    def _analyze_emotional_context(self,
                                 text: str,
                                 cognitive_markers: Dict[str, float]) -> str:
        """Analyze emotional context from text and cognitive markers."""
        # Combine text analysis with cognitive markers
        text_lower = text.lower()
        
        # Emotional indicators in text
        positive_indicators = ['good', 'excellent', 'success', 'positive', 'confident']
        negative_indicators = ['bad', 'failure', 'problem', 'negative', 'worried']
        stress_indicators = ['urgent', 'critical', 'emergency', 'pressure', 'stress']
        
        text_emotion_score = 0
        if any(word in text_lower for word in positive_indicators):
            text_emotion_score += 1
        if any(word in text_lower for word in negative_indicators):
            text_emotion_score -= 1
        if any(word in text_lower for word in stress_indicators):
            text_emotion_score -= 0.5
        
        # Combine with cognitive markers
        stress_level = cognitive_markers.get('stress', 0.5)
        certainty_level = cognitive_markers.get('certainty', 0.5)
        
        # Determine overall emotional context
        if stress_level > 0.7:
            return 'stressed'
        elif certainty_level > 0.7 and text_emotion_score >= 0:
            return 'confident'
        elif certainty_level < 0.3:
            return 'uncertain'
        elif text_emotion_score > 0:
            return 'positive'
        elif text_emotion_score < 0:
            return 'negative'
        else:
            return 'neutral'
    
    def _calculate_interpretation_confidence(self,
                                           neural_output: Dict[str, torch.Tensor],
                                           translation_quality: float,
                                           cognitive_markers: Dict[str, float]) -> float:
        """Calculate overall interpretation confidence."""
        # Neural model confidence
        language_confidence = torch.max(neural_output['language_probs']).item()
        intent_confidence = torch.max(torch.softmax(neural_output['intent_logits'], dim=-1)).item()
        
        # Cognitive state confidence
        certainty = cognitive_markers.get('certainty', 0.5)
        attention = cognitive_markers.get('attention', 0.5)
        
        # Overall confidence calculation
        confidence = (
            0.3 * language_confidence +
            0.2 * intent_confidence +
            0.2 * translation_quality +
            0.15 * certainty +
            0.15 * attention
        )
        
        return min(max(confidence, 0.0), 1.0)
    
    def _assess_context_awareness(self,
                                text: str,
                                context: LanguageContext,
                                cognitive_markers: Dict[str, float]) -> float:
        """Assess how well the interpretation fits the given context."""
        awareness_score = 0.5  # Base score
        
        # Check domain-specific terminology usage
        if context.domain_context == "military":
            military_terms = ['mission', 'target', 'sector', 'hostile', 'friendly']
            term_usage = sum(1 for term in military_terms if term in text.lower())
            awareness_score += 0.1 * min(term_usage / len(military_terms), 1.0)
        
        # Check urgency alignment
        urgency_words = ['urgent', 'critical', 'emergency', 'immediate']
        has_urgency_words = any(word in text.lower() for word in urgency_words)
        if context.urgency_level in ['high', 'critical'] and has_urgency_words:
            awareness_score += 0.2
        elif context.urgency_level in ['low', 'normal'] and not has_urgency_words:
            awareness_score += 0.1
        
        # Check cognitive state alignment
        comprehension = cognitive_markers.get('comprehension', 0.5)
        focus = cognitive_markers.get('focus', 0.5)
        awareness_score += 0.2 * (comprehension + focus) / 2
        
        return min(max(awareness_score, 0.0), 1.0)
    
    def load_model(self, model_path: str):
        """Load pre-trained model weights."""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            self.neural_decoder.load_state_dict(checkpoint['neural_decoder'])
            self.logger.info(f"Neural language interpreter model loaded from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def save_model(self, model_path: str):
        """Save trained model weights."""
        try:
            checkpoint = {
                'neural_decoder': self.neural_decoder.state_dict()
            }
            torch.save(checkpoint, model_path)
            self.logger.info(f"Neural language interpreter model saved to {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            raise

