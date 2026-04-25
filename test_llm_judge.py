"""Test LLM-as-Judge semantic similarity."""

import sys
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, ignore
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.metrics import _semantic_match, _llm_judge_similarity

# Test cases
test_cases = [
    ("না", "নেই", True),  # "No" vs "There isn't" - should match!
    ("নেই", "এর নাম নির্দেশিত হয়নি", True),  # "Not available" vs "Name not specified"
    ("নৌকা", "বাদামি বোটে", True),  # "Boat" vs "In brown boat"
    ("জল", "পানি", True),  # "Water" synonyms
    ("জল পান", "পানি খাওয়া", True),  # "Drinking water"
    ("জল", "পানির দিকে", True),  # "Water" vs "Towards water"
    ("ডানদিকে", "ডান দিকে", True),  # Direction synonyms
    ("বিড়াল", "কুকুর", False),  # Different animals - should NOT match
    ("লাল", "নীল", False),  # Different colors - should NOT match
]

print("=" * 70)
print("Testing LLM-as-Judge Semantic Similarity")
print(f"Judge Model: {os.getenv('JUDGE_MODEL', 'qwen2.5:0.5b (default)')}")
print("=" * 70)

passed = 0
failed = 0

print("\nRunning LLM Judge Tests (this will make API calls):")
print("-" * 70)

for gt, pred, expected in test_cases:
    print(f"\nTesting: '{gt}' vs '{pred}'")
    
    try:
        result = _semantic_match(pred, gt)
        res = _llm_judge_similarity(pred, gt)
        print("result:", result, "expected:", expected)
        print("llm judge result:", res)
        
        if result == expected:
            status = "✓ PASS"
            passed += 1
        else:
            status = "✗ FAIL"
            failed += 1
        
        print(f"  Result: {status} (got {result}, expected {expected})")
        
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        failed += 1

print("\n" + "=" * 70)
print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
print("=" * 70)

if failed == 0:
    print("\n🎉 All tests passed! LLM-as-Judge is working correctly.")
else:
    print(f"\n⚠️  {failed} test(s) failed.")
