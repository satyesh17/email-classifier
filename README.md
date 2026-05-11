# email-classifier

Production-grade email classification with five prompting patterns, Pydantic schema enforcement, and HuggingFace dataset benchmarking. Built as Day 3 of an AI Solutions Architect learning journey.

**What this demonstrates:**
- All 5 LLM prompting patterns implemented as swappable classifiers
- Pydantic + Instructor for guaranteed structured output
- Multi-provider portability (Gemini, Groq, with 5-line provider swaps)
- Real-data benchmarking on Hugging Face's Enron corpus
- Architectural patterns: Strategy pattern, Tool Registry, two-stage validation

---

## The Five Prompting Patterns

Five classifiers, identical interface, different reasoning strategies:

| Classifier | When to Use | Code Complexity |
|---|---|---|
| `ZeroShotClassifier` | Simple well-defined tasks; baseline | ~70 lines |
| `FewShotClassifier` | Tasks needing examples for category boundaries | ~90 lines |
| `ChainOfThoughtClassifier` | Multi-step reasoning, auditable decisions | ~95 lines |
| `ReActClassifier` | Agentic loops with tool use | ~120 lines |
| `StructuredClassifier` | Guaranteed schema output with auto-retry | **~25 lines** |

Each has the same interface:

```python
classifier = StructuredClassifier()
result: EmailClassification = classifier.classify(email_text)
```

The **architectural punchline**: `StructuredClassifier` uses [Instructor](https://python.useinstructor.com/) + [Pydantic](https://pydantic.dev/) to enforce schema validation with automatic retries. Code shrinks 80% (from ~120 → ~25 lines) while reliability *improves*. **Choose the right abstraction; the code writes itself.**

---

## The Pydantic Schema

The contract all five classifiers fulfill:

```python
from typing import Literal
from pydantic import BaseModel, Field


class EmailClassification(BaseModel):
    category: Literal["sales", "support", "billing", "spam", "other"]
    priority: Literal["low", "medium", "high", "urgent"]
    action_required: bool
    summary: str = Field(..., min_length=10, max_length=200)
```

**This is the single source of truth.** It drives the prompt (via `model_json_schema()`), validates LLM output (via `model_validate()`), serializes for storage (via `model_dump_json()`). One definition, four jobs.

---

## Architecture

email-classifier/
├── src/email_classifier/
│   ├── models.py              # Pydantic EmailClassification (the contract)
│   ├── tools.py               # Tools the ReAct agent can call (lookup_sender)
│   └── classifiers/
│       ├── zero_shot.py       # Pattern 1: simplest prompt
│       ├── few_shot.py        # Pattern 2: examples in prompt
│       ├── chain_of_thought.py # Pattern 3: <thinking>...<answer> tags
│       ├── react.py           # Pattern 4: tool use with JSON action format
│       └── structured.py      # Pattern 5: Instructor + Groq Llama 3.1 8B
├── run_benchmark.py            # Real-data benchmark on Enron emails
├── data/sample_classifications.json # Benchmark output (30 emails)
└── pyproject.toml              # Editable install: pip install -e .


**One pattern per file. Strategy pattern across all five.** Adding a 6th pattern is a new file, not a refactor.

---

## Real-Data Benchmark

Tested on 30 emails sampled from [Yale-LILY/aeslc](https://huggingface.co/datasets/Yale-LILY/aeslc) (cleaned Enron corpus) using `StructuredClassifier` (Groq + Llama 3.1 8B Instant).

**Result:** 29/30 (96.7%) schema-valid classifications on first run, against previously-unseen real corporate emails.

### Category Distribution

support: 10   (33%)
sales:    7   (23%)
billing:  6   (20%)
other:    6   (20%)
spam:     0   (0%)   # zero phishing in internal corporate email


**Architectural finding:** Enron's corporate emails (legal redlines, trading ops, internal logistics) don't fit a SaaS customer-support schema cleanly. **Schemas are domain-shaped; "sales/support/billing/spam/other" assumes B2C customer interactions, not internal corporate communication.** Most non-fitting emails were forced into `other` or pattern-matched into the closest category.

### One Failure Worth Studying

One email failed validation across 3 Instructor retries — a forwarded news article ("Enron Article", index 3). The error:

4 validation errors for EmailClassification
priority: Field required [input_value={'properties': {'priority...

**The model returned the JSON Schema instead of an instance.** This failure mode is **schema parroting** — when long, structured input (quoted-printable encoded news, forwarded threads) triggers the model to mirror the schema instead of filling it in.

This was the anchor failure case that drove the Day 3 extension benchmark in [llm-comparator](https://github.com/satyesh17/llm-comparator) — measuring which providers handle this failure mode best.

---

## The Architectural Patterns Built

### 1. Strategy Pattern
Five classifiers, one common method signature (`classify(email_text) -> EmailClassification`). Adding a 6th pattern doesn't touch any existing code. Compare to `LLMProvider` ABC in [llm-comparator](https://github.com/satyesh17/llm-comparator) — same pattern, different domain.

### 2. Tool Registry (in `react.py`)
The model emits JSON-formatted tool calls (`{"tool": "lookup_sender", "args": {...}}`), which dispatch through a name → function dict:

```python
TOOL_REGISTRY = {
    "lookup_sender": lookup_sender,
}

# Generic dispatch — no per-tool branching
tool_fn = TOOL_REGISTRY[action["tool"]]
result = tool_fn(**action["args"])
```

Adding a tool is one line. The orchestration loop never changes.

### 3. Vendor Portability
The `StructuredClassifier` was first written against Gemini, then swapped to Groq when Gemini's daily quota hit. **Five lines changed.** Pydantic schema, prompt template, and all 30-email benchmark code unchanged. **Instructor + Pydantic = vendor-neutral structured outputs.**

### 4. Two-Stage Defense in Depth
The Pydantic schema is embedded in the prompt (LLM tries to comply), then validates the response (rejects non-compliance). When the LLM fails, Instructor automatically re-prompts with the validation error — letting the model see and correct its own mistake.

### 5. Errors as Values (Benchmark Mode)
The benchmark script captures failures with `success: false, error: "..."` rows alongside successes, never crashing on a single bad email. **Robust batch processing beats fail-fast for evaluation jobs.**

---

## Quickstart

```bash
# Clone and install
git clone https://github.com/satyesh17/email-classifier
cd email-classifier
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Setup .env (see .env.example)
cp .env.example .env
# Edit .env to add your GROQ_API_KEY (and optional GOOGLE_API_KEY, HF_TOKEN)

# Try a classifier
python -c "
from email_classifier.classifiers.structured import StructuredClassifier
clf = StructuredClassifier()
print(clf.classify('Hi, my password reset link expired. Please send another.'))
"

# Run the real-data benchmark
python run_benchmark.py
```

Output:

EmailClassification(
category='support', priority='medium',
action_required=True, summary='Password reset link expired'
)

---

## Tech Stack

- **Python 3.13** with `pip install -e .` editable install
- **[Pydantic 2.x](https://pydantic.dev/)** — runtime-validated data models
- **[Instructor](https://python.useinstructor.com/)** by Jason Liu — Pydantic-aware LLM wrapper, ~3M monthly downloads, inspired OpenAI's structured-output feature
- **[Groq](https://groq.com/)** — Llama 3.1 8B Instant on Cerebras LPUs (~14K free RPD)
- **[Google Gemini](https://ai.google.dev/)** Flash-Lite (1000 free RPD)
- **[Hugging Face Datasets](https://huggingface.co/docs/datasets/)** — for [Yale-LILY/aeslc](https://huggingface.co/datasets/Yale-LILY/aeslc) Enron corpus

---

## What I Learned

Three lessons I'd want a future colleague to know after reading this:

**1. The right abstraction reduces code by an order of magnitude.** `StructuredClassifier` (Instructor + Pydantic) is ~25 lines and more reliable than the 120-line manual ReAct implementation. **When defensive parsing dominates your code, you're at the wrong abstraction level.**

**2. Retry-with-backoff cannot fix structural failures.** Daily API quotas, schema parroting on certain input shapes, and category hallucination are all *structural* — retries with the same prompt fail identically. Architects must distinguish transient (rate limits) from structural (quotas, prompt-induced) failures. *See the [llm-comparator schema benchmark](https://github.com/satyesh17/llm-comparator) for measured examples.*

**3. Schemas are domain-shaped.** A schema designed for SaaS customer email doesn't fit Enron's internal corporate email. **Schemas are products of their assumed data distribution.** Real architects design schemas against the data they'll see, not against hypothetical perfect inputs.

---

## Related Projects

- **[llm-comparator](https://github.com/satyesh17/llm-comparator)** — Day 2's multi-provider LLM benchmark, extended on Day 3 with schema-pass-rate measurement across 5 providers using this project's `EmailClassification` schema.

