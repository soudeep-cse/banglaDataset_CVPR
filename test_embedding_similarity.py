"""Test embedding-based semantic similarity."""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the metrics functions
from app.metrics import _compute_similarity, _semantic_match

# Test cases from the user
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
print("Testing Embedding-Based Semantic Similarity")
print("Model: paraphrase-multilingual-MiniLM-L12-v2")
print("=" * 70)

# First, test similarity scores
print("\n1. Similarity Scores (embedding cosine similarity):")
print("-" * 70)
for gt, pred, expected in test_cases[:6]:  # First 6 should be similar
    sim = _compute_similarity(pred, gt)
    status = "✓" if sim >= 0.5 else " "
    print(f"{status} '{gt}' vs '{pred}': {sim:.3f}")

print("\n2. Semantic Match Results (threshold=0.65):")
print("-" * 70)

passed = 0
failed = 0

for gt, pred, expected in test_cases:
    result = _semantic_match(pred, gt, threshold=0.65)
    sim = _compute_similarity(pred, gt)
    
    if result == expected:
        status = "✓ PASS"
        passed += 1
    else:
        status = "✗ FAIL"
        failed += 1
    
    print(f"{status} | '{gt}' vs '{pred}'")
    print(f"       Similarity: {sim:.3f}, Match: {result}, Expected: {expected}")
    print()

print("=" * 70)
print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
print("=" * 70)

if failed == 0:
    print("\n🎉 All tests passed! Embedding-based similarity is working correctly.")
else:
    print(f"\n⚠️  {failed} test(s) failed. You may need to adjust the threshold.")
