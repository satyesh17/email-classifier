"""ReAct (Reasoning + Acting) classifier with JSON-formatted tool calls."""

import json
import os
import re

import google.generativeai as genai
from dotenv import load_dotenv

from ...email_classifier.models import EmailClassification
from ...email_classifier.tools import lookup_sender

load_dotenv()


# Tool registry — maps tool names to callables
TOOL_REGISTRY = {
    "lookup_sender": lookup_sender,
}


PROMPT_TEMPLATE = """You are an email classification system with access to tools.

You produce output by alternating between:
- Thought: <your reasoning>
- Action: <one JSON action object>
- Observation: <result; the system fills this in>

Available tools:
- lookup_sender — Args: {{"email": "string"}}. Returns sender info from the customer database.

When you have enough information, emit:
- Action: {{"tool": "finish"}}
- Final Answer: <JSON matching the schema below>

Required output schema:
{schema}

EXAMPLE TRACE:
EMAIL: I need help with my login from alice@example.com
Thought: I should check if Alice is a known customer.
Action: {{"tool": "lookup_sender", "args": {{"email": "alice@example.com"}}}}
Observation: {{"is_existing_customer": true, "tier": "enterprise", "open_tickets": 2}}
Thought: Alice is an enterprise customer. This is a support request.
Action: {{"tool": "finish"}}
Final Answer: {{"category": "support", "priority": "high", "action_required": true, "summary": "Enterprise customer Alice requests login help."}}

Now classify the email below. Stop and emit only ONE thought + action at a time. Wait for the Observation before continuing.

EMAIL: {email_text}
"""


class ReActClassifier:
    """Classifies emails using a ReAct loop with JSON tool calls."""

    name = "react"
    MAX_ITERATIONS = 5

    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found.")
        
        genai.configure(api_key=api_key)
        self.model = model
        self._model = genai.GenerativeModel(model)

    def _build_initial_prompt(self, email_text: str) -> str:
        schema = json.dumps(EmailClassification.model_json_schema(), indent=2)
        return PROMPT_TEMPLATE.format(schema=schema, email_text=email_text)

    def _extract_action_block(self, response_text: str) -> str | None:
        """Find Action: {...} and return the JSON, balancing braces correctly."""
        # Find where 'Action:' appears, followed by a {
        action_marker = re.search(r"Action:\s*\{", response_text)
        if not action_marker:
            return None
        
        # Start at the opening brace
        start = action_marker.end() - 1
        
        # Walk forward, counting braces until balanced
        depth = 0
        for i in range(start, len(response_text)):
            if response_text[i] == "{":
                depth += 1
            elif response_text[i] == "}":
                depth -= 1
                if depth == 0:
                    return response_text[start:i + 1]   # found end
        
        return None   # unbalanced; bail

    def _execute_action(self, action_obj: dict) -> str:
        """Dispatch to a tool from the registry. Returns observation as JSON string."""
        tool_name = action_obj.get("tool")
        args = action_obj.get("args", {})
        
        if tool_name not in TOOL_REGISTRY:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        
        tool_fn = TOOL_REGISTRY[tool_name]
        try:
            result = tool_fn(**args)
            return json.dumps(result)
        except Exception as e:
            return json.dumps({"error": f"Tool execution failed: {e}"})

    def _parse_final_answer(self, response_text: str) -> EmailClassification:
        """Extract Final Answer JSON from the response and validate against schema."""
        match = re.search(r"Final Answer:\s*(\{.*\})", response_text, re.DOTALL)
        if not match:
            raise ValueError(f"Final Answer not found in: {response_text[:300]}")
        
        json_text = match.group(1).strip()
        
        # Strip Markdown fences if present
        if json_text.startswith("```"):
            json_text = "\n".join(json_text.split("\n")[1:-1])
        
        parsed = json.loads(json_text)
        return EmailClassification.model_validate(parsed)

    def classify(self, email_text: str) -> EmailClassification:
        """Run the ReAct loop. Returns validated EmailClassification or raises."""
        conversation = self._build_initial_prompt(email_text)
        
        for iteration in range(self.MAX_ITERATIONS):
            response = self._model.generate_content(conversation)
            response_text = response.text.strip()
            
            # Append the model's response to the running transcript
            conversation += "\n" + response_text
            
            # Did the model finish?
            if "Final Answer:" in response_text:
                return self._parse_final_answer(response_text)
            
            # Otherwise, parse the action JSON and run the tool
            action_text = self._extract_action_block(response_text)
            if not action_text:
                raise ValueError(
                    f"No Action JSON found in iteration {iteration}. "
                    f"Response: {response_text[:300]}"
                )
            
            try:
                action_obj = json.loads(action_text)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Action JSON malformed in iteration {iteration}: {action_text[:200]}"
                ) from e
            
            # Special case: model said "finish" but didn't emit Final Answer yet
            if action_obj.get("tool") == "finish":
                # Re-prompt: the model needs to also emit Final Answer
                conversation += "\nObservation: Please now emit the Final Answer JSON.\n"
                continue
            
            observation = self._execute_action(action_obj)
            conversation += f"\nObservation: {observation}\n"
        
        raise RuntimeError(
            f"ReAct loop did not finish after {self.MAX_ITERATIONS} iterations."
        )