from ..models import EmailClassification

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

import os, json
from dotenv import load_dotenv

load_dotenv() 

PROMPT_TEMPLATE = """
        You are an email classification system. Analyze the email below and return a JSON object matching this schema:
        {schema}
        
        Email:
        {email_text}
        
        Return ONLY the JSON object, no preamble or explanation.
        """

class ZeroShotClassifier:

    name="zero_shot"

    def __init__(self, model= "gemini-2.5-flash-lite"):

        api_key=os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        
        genai.configure(api_key=api_key)
        self.model=model
        self._model=genai.GenerativeModel(model)


    def build_prompt(self, email_text:str) -> str:

        schema=json.dumps(EmailClassification.model_json_schema(), indent=2)
        return PROMPT_TEMPLATE.format(schema=schema, email_text=email_text)

    def classify(self, email_text:str) -> EmailClassification:
        """Classify an email - raises if model return malfored output"""

        
        prompt=self.build_prompt(email_text)
        response = self._model.generate_content(prompt)
        response_text = response.text.strip()

        if response_text.startswith("```"):
            response_text = "\n".join(response_text.split("\n")[1:-1])

        try:
            parsed=json.loads(response_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Model retuend invalid JSON. Raw response (first 200 chars):\n{response_text[:200]}") from e
        
        return EmailClassification.model_validate(parsed)

            
    


     