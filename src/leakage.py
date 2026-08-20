"""
Campaign-level leakage checks, used by every stage that touches a train/test
split. This is the single most load-bearing piece of code in the whole
project for defensibility: the thesis's central promise is that no attack
campaign appears on both sides of any split, and every claim of that must be
backed by this actually being checked in code, not just asserted in prose.
"""

def assert_no_campaign_leakage(train_campaign_ids, test_campaign_ids, context=""):
    """
    Raises AssertionError if any campaign_id appears in both sets.

    train_campaign_ids / test_campaign_ids: any iterable of campaign_id
    strings (e.g. a pandas Series column).
    """
    train_set = set(train_campaign_ids)
    test_set = set(test_campaign_ids)
    overlap = train_set & test_set
    if overlap:
        raise AssertionError(
            f"CAMPAIGN LEAKAGE DETECTED{' (' + context + ')' if context else ''}: "
            f"{len(overlap)} campaign_id(s) appear in both train and test: "
            f"{sorted(overlap)[:10]}{' ...' if len(overlap) > 10 else ''}"
        )
    return {
        "train_campaigns": len(train_set),
        "test_campaigns": len(test_set),
        "overlap": 0,
    }


def assert_dataset_is_test_only(df, source_dataset_col, allowed_test_only_sources, context=""):
    """
    Stronger guarantee for DARPA2000: assert that a given training dataframe
    contains ZERO rows from any source_dataset name that is designated
    test-only (e.g. "darpa2000_lldos1.0", "darpa2000_lldos2.0.2"). This
    enforces the project rule that DARPA2000 is used only for the final
    Stage 5 evaluation and never touches any training or hyperparameter
    tuning code.
    """
    present = set(df[source_dataset_col].unique())
    violation = present & set(allowed_test_only_sources)
    if violation:
        raise AssertionError(
            f"TEST-ONLY DATASET LEAKED INTO TRAINING{' (' + context + ')' if context else ''}: "
            f"found rows from {sorted(violation)} in a training dataframe"
        )
    return {"checked_sources": sorted(present), "violation": None}
