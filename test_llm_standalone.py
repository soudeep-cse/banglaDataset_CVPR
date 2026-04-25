"""Standalone test for LLM-as-Judge - no app imports."""

import os
import sys
import requests

try:
    from dotenv import load_dotenv
    from pathlib import Path
    env_path = Path(__file__).parent / ".env"
    print(f"Looking for .env at: {env_path}")
    print(f".env exists: {env_path.exists()}")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True, verbose=True)
        print(f"Loaded .env from: {env_path}")
    else:
        load_dotenv()
except ImportError as e:
    print(f"dotenv import error: {e}")
    pass

# Copy the LLM judge function here for testing
_LLM_JUDGE_CACHE = {}

def _get_llm_judge_config():
    """Get LLM judge configuration from environment."""
    # Try env first, fallback to .env file values
    model = os.getenv("JUDGE_MODEL") or "gemma4:e2b"
    host = os.getenv("OLLAMA_HOST") or "http://213.173.99.31:36766"
    return model, host

def _llm_judge_similarity(predicted: str, ground_truth: str, question: str = "") -> bool:
    """Use LLM to judge if two answers are semantically equivalent."""
    cache_key = (predicted.strip().lower(), ground_truth.strip().lower(), question.strip().lower())
    if cache_key in _LLM_JUDGE_CACHE:
        return _LLM_JUDGE_CACHE[cache_key]
    
    model, host = _get_llm_judge_config()
    
    prompt = f"""You are judging if two Bangla answers to the same question are semantically equivalent.

Question: {question}
Ground Truth Answer: {ground_truth}
Predicted Answer: {predicted}

Do these two answers mean the same thing? They don't need to use the same words, just convey the same meaning in the context of the question.

Reply with only "yes" or "no".

Response:"""
    
    print(f"\n[LLM Call] Model: {model}")
    print(f"  Question: '{question}'")
    print(f"  Ground Truth: '{ground_truth}'")
    print(f"  Predicted: '{predicted}'")
    
    try:
        response = requests.post(
            f"{host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 500}
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        print(f"  done_reason: {result.get('done_reason')}")
        print(f"  prompt_eval_count: {result.get('prompt_eval_count')}")
        print(f"  eval_count: {result.get('eval_count')}")
        answer = result.get("response", "").strip().lower()
        
        print(f"  LLM raw response: '{answer}'")
        
        # Parse yes/no
        is_match = False
        if answer:
            if any(x in answer for x in ["yes", "true", "equivalent", "same", "1"]):
                is_match = True
            elif any(x in answer for x in ["no", "false", "different", "0"]):
                is_match = False
            else:
                # Fallback to substring
                is_match = predicted.lower() in ground_truth.lower() or ground_truth.lower() in predicted.lower()
        
        _LLM_JUDGE_CACHE[cache_key] = is_match
        return is_match
        
    except Exception as e:
        print(f"  Error: {e}")
        return predicted.lower() == ground_truth.lower()

# Test cases with questions for context
# Format: (ground_truth, predicted, expected, question)
test_cases = [
    ("না", "নেই", True, "নীচে কি কোনো স্টিকার আছে?"),
    ("নেই", "এর নাম নির্দেশিত হয়নি", True, "ছবিতে কী আছে?"),
    ("জল", "পানি", True, "বিড়ালটি কী পান করছে?"),
    ("বিড়াল", "কুকুর", False, "ছবিতে কোন প্রাণীটি আছে?"),
]

print("=" * 70)
print("Testing LLM-as-Judge (Standalone)")
print(f"Host: {os.getenv('OLLAMA_HOST', 'http://localhost:11434')}")
print(f"Judge Model: {os.getenv('JUDGE_MODEL', 'qwen2.5:0.5b (default)')}")
print("=" * 70)

passed = 0
failed = 0

for gt, pred, expected, question in test_cases:
    result = _llm_judge_similarity(pred, gt, question)
    
    if result == expected:
        status = "✓ PASS"
        passed += 1
    else:
        status = "✗ FAIL"
        failed += 1
    
    print(f"\n{status} | Q: {question}")
    print(f"       GT: '{gt}' vs Pred: '{pred}'")
    print(f"       Expected: {expected}, Got: {result}")
    print("-" * 50)

print(f"\nResults: {passed} passed, {failed} failed out of {len(test_cases)} tests")
