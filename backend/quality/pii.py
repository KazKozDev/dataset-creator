from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from typing import Dict, Any, List

# PII types that are actually sensitive and should be detected
# Excludes DATE_TIME, NRP (nationality/religion/political), URL which cause many false positives
SENSITIVE_PII_TYPES = {
    'EMAIL_ADDRESS',
    'PHONE_NUMBER', 
    'CREDIT_CARD',
    'IBAN_CODE',
    'IP_ADDRESS',
    'PERSON',
    'LOCATION',
    'MEDICAL_LICENSE',
    'US_SSN',
    'US_PASSPORT',
    'US_DRIVER_LICENSE',
    'US_BANK_NUMBER',
    'UK_NHS',
    'SG_NRIC_FIN',
    'AU_ABN',
    'AU_ACN',
    'AU_TFN',
    'AU_MEDICARE',
}

# Minimum confidence score to consider a detection valid
MIN_CONFIDENCE_SCORE = 0.7

class PIIAnalyzer:
    def __init__(self):
        # Initialize engines lazily
        self._analyzer = None
        self._anonymizer = None

    @property
    def analyzer(self):
        if self._analyzer is None:
            self._analyzer = AnalyzerEngine()
        return self._analyzer

    @property
    def anonymizer(self):
        if self._anonymizer is None:
            self._anonymizer = AnonymizerEngine()
        return self._anonymizer

    def analyze(self, text: str, score_threshold: float = MIN_CONFIDENCE_SCORE) -> List[Dict[str, Any]]:
        """
        Analyze text for PII.
        Returns a list of detected entities with high confidence.
        Filters out low-confidence detections and non-sensitive entity types.
        """
        if not text:
            return []
            
        results = self.analyzer.analyze(
            text=text, 
            language='en',
            score_threshold=score_threshold
        )
        
        # Filter by sensitive PII types and minimum score
        filtered_results = [
            {
                'type': res.entity_type,
                'start': res.start,
                'end': res.end,
                'score': res.score
            }
            for res in results
            if res.entity_type in SENSITIVE_PII_TYPES and res.score >= score_threshold
        ]
        
        return filtered_results

    def anonymize(self, text: str, score_threshold: float = MIN_CONFIDENCE_SCORE) -> str:
        """
        Anonymize PII in text.
        Only anonymizes high-confidence sensitive PII.
        """
        if not text:
            return ""
            
        results = self.analyzer.analyze(
            text=text, 
            language='en',
            score_threshold=score_threshold
        )
        
        # Filter to only sensitive PII types
        filtered_results = [
            res for res in results
            if res.entity_type in SENSITIVE_PII_TYPES and res.score >= score_threshold
        ]
        
        if not filtered_results:
            return text
            
        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=filtered_results
        )
        return anonymized_result.text

# Global instance
pii_analyzer = PIIAnalyzer()
