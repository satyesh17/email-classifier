"""Structured-output classifier using Instructor + Gemini's native function calling."""

import os

import instructor
import google.generativeai as genai
from dotenv import load_dotenv

from ...email_classifier.models import EmailClassification

load_dotenv()


class StructuredClassifier:
    """Classifies emails with Pydantic-enforced structured output via Instructor."""

    name = "structured"
    MAX_RETRIES = 3

    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found.")
        
        genai.configure(api_key=api_key)
        self.model = model
        
        # Wrap a raw Gemini model with Instructor to make it Pydantic-aware
        self._client = instructor.from_gemini(
            client=genai.GenerativeModel(model),
            mode=instructor.Mode.GEMINI_JSON,
        )

    def classify(self, email_text: str) -> EmailClassification:
        """Classify an email — Instructor handles schema enforcement and retries."""
        prompt = f"Classify the following email into the structured schema.\n\nEMAIL:\n{email_text}"
        
        return self._client.messages.create(
            messages=[{"role": "user", "content": prompt}],
            response_model=EmailClassification,
            max_retries=self.MAX_RETRIES,
        )