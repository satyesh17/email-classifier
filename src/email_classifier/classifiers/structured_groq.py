"""Structured-output classifier using Instructor + Groq."""

import os

import instructor
from groq import Groq
from dotenv import load_dotenv

from ...email_classifier.models import EmailClassification

load_dotenv()


class StructuredClassifier:
    """Classifies emails with Pydantic-enforced structured output via Instructor + Groq."""

    name = "structured"
    MAX_RETRIES = 3

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found.")
        
        self.model = model
        # Wrap a raw Groq client with Instructor
        self._client = instructor.from_groq(
            client=Groq(api_key=api_key),
            mode=instructor.Mode.JSON,
        )

    def classify(self, email_text: str) -> EmailClassification:
        """Classify an email — Instructor handles schema enforcement and retries."""
        prompt = f"Classify the following email into the structured schema.\n\nEMAIL:\n{email_text}"
        
        return self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_model=EmailClassification,
            max_retries=self.MAX_RETRIES,
        )