from detoxify import Detoxify
from typing import Dict, Any

class ToxicityAnalyzer:
    def __init__(self):
        # Load model lazily to save memory if not used
        self._model = None

    @property
    def model(self):
        if self._model is None:
            # Using 'original' model which is smaller but effective
            self._model = Detoxify('original')
        return self._model

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze text for toxicity.
        Returns a dictionary with scores for:
        - toxicity
        - severe_toxicity
        - obscene
        - threat
        - insult
        - identity_hate
        """
        if not text:
            return {
                'toxicity': 0.0,
                'severe_toxicity': 0.0,
                'obscene': 0.0,
                'threat': 0.0,
                'insult': 0.0,
                'identity_hate': 0.0
            }
            
        results = self.model.predict(text)
        # Convert numpy floats to python floats
        return {k: float(v) for k, v in results.items()}

# Global instance
toxicity_analyzer = ToxicityAnalyzer()
