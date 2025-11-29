import pandas as pd
import numpy as np

def dataset_split(
    df: pd.DataFrame,
    user_col: str = "user",
    item_col: str = "item",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
):
    """
    Split interactions into train/val/test such that:
      - Items do NOT overlap between splits.
      - Row counts roughly follow the given ratios.
      - Best-effort: each user with >= 3 distinct items gets at least
        one interaction in each split.
    """
    splits = ["train", "val", "test"]

    # ---- 1) Check ratios ----
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    # ---- 2) Per-item row counts ----
    item_counts = df[item_col].value_counts().to_dict()
    items = list(item_counts.keys())

    rng = np.random.default_rng(random_state)
    rng.shuffle(items)

    total_rows = len(df)
    targets = {
        "train": total_rows * train_ratio,
        "val":   total_rows * val_ratio,
        "test":  total_rows * test_ratio,
    }
    split_rows = {s: 0 for s in splits}
    item_to_split = {}

    # ---- 3) Initial greedy assignment (row-based) ----
    for it in items:
        remaining = {s: targets[s] - split_rows[s] for s in splits}
        best_split = max(remaining, key=remaining.get)
        item_to_split[it] = best_split
        split_rows[best_split] += item_counts[it]

    # ---- 4) Ensure each split has at least one item (if possible) ----
    for s in splits:
        has_item = any(item_to_split[it] == s for it in items)
        if not has_item:
            donor = max(splits, key=lambda x: split_rows[x])
            donor_items = [it for it in items if item_to_split[it] == donor]
            move_item = min(donor_items, key=lambda it: item_counts[it])
            item_to_split[move_item] = s
            split_rows[donor] -= item_counts[move_item]
            split_rows[s] += item_counts[move_item]

    # ---- 5) Best-effort per-user coverage ----
    n_splits = len(splits)

    # user → list of items
    user_items_map = (
        df.groupby(user_col)[item_col]
        .apply(lambda x: x.unique().tolist())
        .to_dict()
    )

    for uid, u_items in user_items_map.items():
        # If user has fewer items than splits, cannot cover all splits
        if len(u_items) < n_splits:
            continue

        # Count this user's items per split
        user_split_counts = {s: 0 for s in splits}
        for it in u_items:
            s = item_to_split[it]
            user_split_counts[s] += 1

        missing_splits = [s for s in splits if user_split_counts[s] == 0]
        if not missing_splits:
            continue

        for s_missing in missing_splits:
            # we can only move from splits where user has >1 items
            candidate_items = [
                it for it in u_items
                if user_split_counts[item_to_split[it]] > 1
            ]
            if not candidate_items:
                break  # can't fix this user further

            # move the smallest item by count
            cand = min(candidate_items, key=lambda it: item_counts[it])
            old_split = item_to_split[cand]

            item_to_split[cand] = s_missing
            split_rows[old_split] -= item_counts[cand]
            split_rows[s_missing] += item_counts[cand]

            user_split_counts[old_split] -= 1
            user_split_counts[s_missing] += 1

    # ---- 6) Build final DataFrames ----
    item_split_series = df[item_col].map(item_to_split)

    train_df = df[item_split_series == "train"].reset_index(drop=True)
    val_df   = df[item_split_series == "val"].reset_index(drop=True)
    test_df  = df[item_split_series == "test"].reset_index(drop=True)

def write_split_txt(
    df: pd.DataFrame,
    filepath: str,
    user_col: str = "user",
    item_col: str = "item",
):
    # group items per user (unique, in order of appearance)
    grouped = (
        df.groupby(user_col)[item_col]
        .apply(lambda x: x.unique().tolist())
        .sort_index()  # users in ascending order
    )

    with open(filepath, "w") as f:
        for uid, items in grouped.items():
            items_str = " ".join(str(it) for it in items)
            line = f"{uid-1} {items_str}\n"
            f.write(line)

    return train_df, val_df, test_df
