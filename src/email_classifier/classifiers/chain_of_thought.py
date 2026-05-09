"""Few-shot email classifier using Gemini."""

import json
import os
import re

import google.generativeai as genai
from dotenv import load_dotenv

from ..models import EmailClassification

load_dotenv()

EXAMPLES = [
    {
        "email": "Hi, my credit has expired for $2000 and I cannot proceed unless that is reinstated. Please fix this urgently.",
        "classification": {
            "category": "support",
            "priority": "urgent",
            "action_required": True,
            "summary": "User requests a reimbursement for expired credits."
        }
    },
    {
        "email": "Hi, my password reset link expired. Can you send a new one?",
        "classification": {
            "category": "support",
            "priority": "medium",
            "action_required": True,
            "summary": "User requests a fresh password reset link after the previous one expired."
        }
    },
    {
        "email": "Hello, I'm evaluating your enterprise plan for a team of 200. Could we set up a call next week?",
        "classification": {
            "category": "sales",
            "priority": "high",
            "action_required": True,
            "summary": "Prospect requests a sales call to evaluate the enterprise plan for 200 users."
        }
    },
    {
        "email": "I noticed two charges of $49 on my credit card statement for October. Can you confirm and refund the duplicate?",
        "classification": {
            "category": "billing",
            "priority": "high",
            "action_required": True,
            "summary": "Customer reports a duplicate charge of $49 in October and requests a refund."
        }
    },
    {
        "email": "Congratulations! You have been selected for $10 million prize. Click here to claim now: http://suspicious-link.example",
        "classification": {
            "category": "spam",
            "priority": "low",
            "action_required": False,
            "summary": "Phishing email claiming a prize, with a suspicious external link."
        }
    },
    {
        "email": "Thanks for the awesome support last week, just wanted to say my team is really happy.",
        "classification": {
            "category": "other",
            "priority": "low",
            "action_required": False,
            "summary": "Customer expresses gratitude for recent support; no action required."
        }
    },
]

PROMPT_TEMPLATE = """You are an email classification system. Classify emails into JSON matching this schema:

{schema}

Here are examples of correct classifications:

{examples}

Now classify the email below. Think through it step by step before answering.

REASONING STEPS:
1. What is the sender's primary intent?
2. Which category best fits that intent?
3. How urgent is the response needed?
4. Does this require an action, or is it informational?

EMAIL: {email_text}

Wrap your reasoning in <thinking>...</thinking> tags.
Then output ONLY the JSON classification wrapped in <answer>...</answer> tags.
"""

def _format_example(example: dict, index: int) -> str:
    """Format a single example as 'EMAIL N: ... \\n CLASSIFICATION N: ...'"""
    email = example["email"]
    classification_json = json.dumps(example["classification"])
    return f"EMAIL {index}: {email}\nCLASSIFICATION {index}: {classification_json}"

class ChainOfThoughtClassifier:
    """Classifies emails using few-shot prompting via Gemini."""

    name = "chain_of_thought"

    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found.")
        
        genai.configure(api_key=api_key)
        self.model = model
        self._model = genai.GenerativeModel(model)

    def _build_prompt(self, email_text: str) -> str:
        schema = json.dumps(EmailClassification.model_json_schema(), indent=2)
        
        examples_text = "\n\n".join(
            _format_example(ex, i + 1) 
            for i, ex in enumerate(EXAMPLES)
        )
        
        return PROMPT_TEMPLATE.format(
            schema=schema,
            examples=examples_text,
            email_text=email_text,
        )
    
    def classify(self, email_text: str) -> EmailClassification:
        """Classify with chain-of-thought reasoning. Raises on malformed output."""
        prompt = self._build_prompt(email_text)
        response = self._model.generate_content(prompt)
        
        response_text = response.text.strip()
        
        # Extract <answer>...</answer> content if present
        match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
        if match:
            response_text = match.group(1).strip()
        
        # Strip Markdown fences if present
        if response_text.startswith("```"):
            response_text = "\n".join(response_text.split("\n")[1:-1])
        
        # Parse JSON
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Model returned invalid JSON. Raw response (first 200 chars):\n{response_text[:200]}"
            ) from e
    
        return EmailClassification.model_validate(parsed)