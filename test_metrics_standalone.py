"""Standalone test for semantic similarity functions."""

# Copy the relevant functions here for testing
from typing import Any

# Bangla synonym mappings for semantic similarity
BANGLA_SYNONYMS = {
    # Water-related
    "জল": ["পানি", "জল"],
    "পানি": ["জল", "পানি"],
    # Boat-related
    "নৌকা": ["বোট", "নৌকা"],
    "বোট": ["নৌকা", "বোট"],
    # Direction-related
    "ডানদিকে": ["ডান দিকে", "ডানদিকে", "ডানে"],
    "ডান দিকে": ["ডানদিকে", "ডান দিকে", "ডানে"],
    "বাঁদিকে": ["বাঁ দিকে", "বাঁদিকে", "বাঁয়ে"],
    "বাঁ দিকে": ["বাঁদিকে", "বাঁ দিকে", "বাঁয়ে"],
    # None/Not available
    "নেই": ["নির্দেশিত হয়নি", "নেই", "নয়", "নাই"],
    # Drink/eat
    "পান": ["খাওয়া", "পান"],
    "খাওয়া": ["পান", "খাওয়া"],
}


def _get_synonym_set(word: str) -> set[str]:
    """Get all synonyms for a word including the word itself."""
    word = word.strip()
    if word in BANGLA_SYNONYMS:
        return set(BANGLA_SYNONYMS[word])
    return {word}


def _tokens(text: str) -> set[str]:
    """Extract tokens from text for overlap calculation."""
    words = text.strip().split()
    tokens = []
    for word in words:
        clean = word.strip("।.,!?;:()")
        if clean:
            tokens.append(clean)
    return set(tokens)


def _semantic_match(predicted: str, ground_truth: str, threshold: float = 0.4) -> bool:
    """
    Check if predicted semantically matches ground truth.
    """
    norm_pred = predicted.strip().lower()
    norm_truth = ground_truth.strip().lower()

    # Strategy 1: Direct match
    if norm_pred == norm_truth:
        return True

    # Strategy 2: Token overlap with synonym expansion
    pred_tokens = _tokens(norm_pred)
    truth_tokens = _tokens(norm_truth)

    if not pred_tokens or not truth_tokens:
        return norm_pred == norm_truth

    # Expand tokens with synonyms
    pred_expanded = set()
    for token in pred_tokens:
        pred_expanded.update(_get_synonym_set(token))

    truth_expanded = set()
    for token in truth_tokens:
        truth_expanded.update(_get_synonym_set(token))

    # Calculate overlap with synonym expansion
    intersection = pred_expanded & truth_expanded
    union = pred_expanded | truth_expanded

    if not union:
        return False

    overlap = len(intersection) / len(union)
    print(f"    Tokens: pred={pred_tokens}, truth={truth_tokens}")
    print(f"    Expanded: pred={pred_expanded}, truth={truth_expanded}")
    print(f"    Overlap: {len(intersection)}/{len(union)} = {overlap:.2f}")

    if overlap >= threshold:
        return True

    # Strategy 3: Substring match
    if norm_pred in norm_truth or norm_truth in norm_pred:
        return True

    # Strategy 4: Any synonym overlap
    for p_token in pred_expanded:
        for t_token in truth_expanded:
            if p_token == t_token:
                return True
            if len(p_token) > 2 and len(t_token) > 2:
                if p_token in t_token or t_token in p_token:
                    return True

    return False


# Test cases from the user
test_cases = [
    ("নেই", "এর নাম নির্দেশিত হয়নি", True),
    ("নৌকা", "বাদামি বোটে", True),
    ("জল", "পানি", True),
    ("জল পান", "পানি খাওয়া", True),
    ("জল", "পানির দিকে", True),
    ("ডানদিকে", "ডান দিকে", True),
]

print("=" * 60)
print("Testing Semantic Similarity for Bangla-Bayanno")
print("=" * 60)

for gt, pred, expected in test_cases:
    print(f"\nTest: '{gt}' vs '{pred}'")
    result = _semantic_match(pred, gt)
    status = "✓ PASS" if result == expected else "✗ FAIL"
    print(f"Result: {status} (got {result}, expected {expected})")

print("\n" + "=" * 60)
