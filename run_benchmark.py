"""Run StructuredClassifier on a sample of Enron emails from HuggingFace."""

import json
import time
from pathlib import Path

from datasets import load_dataset

from src.email_classifier.classifiers.structured_groq import StructuredClassifier


# Config — tune these as you go
SAMPLE_SIZE = 30
RANDOM_SEED = 42
OUTPUT_PATH = Path("data/sample_classifications.json")


def main():
    # 1. Load dataset
    print(f"Loading Yale-LILY/aeslc...")
    dataset = load_dataset("Yale-LILY/aeslc", split="train")
    print(f"  ✅ Loaded {len(dataset):,} emails")
    
    # 2. Shuffle + sample
    sample = dataset.shuffle(seed=RANDOM_SEED).select(range(SAMPLE_SIZE))
    print(f"  Sampling {SAMPLE_SIZE} emails (seed={RANDOM_SEED})")
    
    # 3. Setup classifier
    print(f"\nInitializing StructuredClassifier (Groq + Llama)...")
    clf = StructuredClassifier()
    
    # 4. Run classification
    print(f"\nClassifying {SAMPLE_SIZE} emails...\n")
    results = []
    
    for i, item in enumerate(sample):
        email_text = item["email_body"]
        subject = item["subject_line"]
        
        # Some emails may be empty or nearly so — skip
        if not email_text or len(email_text.strip()) < 10:
            print(f"  [{i+1}/{SAMPLE_SIZE}] SKIP — too short")
            continue
        
        try:
            start = time.perf_counter()
            classification = clf.classify(email_text)
            latency_ms = (time.perf_counter() - start) * 1000
            
            results.append({
                "index": i,
                "subject": subject,
                "email_body_preview": email_text[:200],
                "classification": classification.model_dump(),
                "latency_ms": round(latency_ms, 1),
                "success": True,
                "error": None,
            })
            
            print(f"  [{i+1}/{SAMPLE_SIZE}] ✅ {classification.category}/{classification.priority} ({latency_ms:.0f}ms)")
            
        except Exception as e:
            results.append({
                "index": i,
                "subject": subject,
                "email_body_preview": email_text[:200],
                "classification": None,
                "latency_ms": None,
                "success": False,
                "error": f"{type(e).__name__}: {str(e)[:200]}",
            })
            
            print(f"  [{i+1}/{SAMPLE_SIZE}] ❌ {type(e).__name__}: {str(e)[:80]}")
    
    # 5. Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n✅ Saved {len(results)} results to {OUTPUT_PATH}")
    
    # 6. Quick summary stats
    successes = sum(1 for r in results if r["success"])
    failures = len(results) - successes
    
    print(f"\n=== Summary ===")
    print(f"  Total processed:  {len(results)}")
    print(f"  Successes:        {successes}  ({successes/len(results)*100:.1f}%)")
    print(f"  Failures:         {failures}  ({failures/len(results)*100:.1f}%)")
    
    if successes > 0:
        # Category distribution
        from collections import Counter
        categories = Counter(
            r["classification"]["category"] for r in results if r["success"]
        )
        print(f"\n  Category distribution:")
        for cat, count in categories.most_common():
            print(f"    {cat}: {count}")


if __name__ == "__main__":
    main()