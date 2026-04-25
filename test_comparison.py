"""Test the improved semantic similarity comparison."""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import only the metrics module functions we need
from app.metrics import compare_answers, _semantic_match

# User's examples
test_cases = [
    # (ground_truth, predicted, expected)
    ("নেই", "এর নাম নির্দেশিত হয়নি", True),  # "Not available" vs "Name not specified"
    ("নৌকা", "বাদামি বোটে", True),  # "Boat" vs "In brown boat"
    ("জল", "পানি", True),  # "Water" synonyms
    ("জল পান", "পানি খাওয়া", True),  # "Drinking water"
    ("জল", "পানির দিকে", True),  # "Water" vs "Towards water"
    ("ডানদিকে", "ডান দিকে", True),  # Direction synonyms
]

print("Testing semantic similarity:\n")

for gt, pred, expected in test_cases:
    result = compare_answers(pred, gt, "descriptive")
    status = "✓ PASS" if result == expected else "✗ FAIL"
    print(f"{status}")
    print(f"  Ground Truth: {gt}")
    print(f"  Predicted:    {pred}")
    print(f"  Match: {result} (expected: {expected})")
    print()

# Test semantic match function directly
print("\nDirect _semantic_match tests:")
direct_tests = [
    ("নেই", "এর নাম নির্দেশিত হয়নি"),
    ("নৌকা", "বাদামি বোটে"),
    ("জল", "পানি"),
    ("জল পান", "পানি খাওয়া"),
    ("জল", "পানির দিকে"),
    ("ডানদিকে", "ডান দিকে"),
]

for gt, pred in direct_tests:
    match = _semantic_match(pred, gt)
    print(f"  '{gt}' vs '{pred}': {match}")
