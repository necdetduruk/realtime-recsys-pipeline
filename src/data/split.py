"""
Time-based train/val/test split for H&M transactions.

Split strategy:
- Train: < 2020-07-01  (~89% of data)
- Val:   2020-07-01 → 2020-08-01  (~4.3%)
- Test:  2020-08-01 → 2020-09-08  (~4.7%)

Memory-efficient implementation: split first, then deduplicate within
each partition separately to avoid sorting the full 31M row dataset.
"""

import gc
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

TRAIN_END = pd.Timestamp("2020-07-01")
VAL_END = pd.Timestamp("2020-08-01")
TEST_END = pd.Timestamp("2020-09-08")

DTYPE_MAP = {
    "customer_id": "string",
    "article_id": "string",
    "price": "float32",
}


def dedup_keep_last(df: pd.DataFrame) -> pd.DataFrame:
    """Keep most recent purchase per user-article pair."""
    return (
        df.sort_values("t_dat")
        .drop_duplicates(subset=["customer_id", "article_id"], keep="last")
        .reset_index(drop=True)
    )


def process_and_save(df: pd.DataFrame, name: str) -> None:
    print(f"  Deduplicating {name}...")
    df = dedup_keep_last(df)
    print(f"  {name}: {len(df):,} rows after dedup")
    out_path = PROCESSED_DIR / f"{name}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading transactions (chunked)...")
    chunk_size = 2_000_000

    reader = pd.read_csv(
        RAW_DIR / "transactions_train.csv",
        usecols=["t_dat", "customer_id", "article_id", "price"],
        dtype=DTYPE_MAP,
        parse_dates=["t_dat"],
        chunksize=chunk_size,
    )

    train_chunks, val_chunks, test_chunks = [], [], []

    for i, chunk in enumerate(reader):
        chunk = chunk[chunk["t_dat"] < TEST_END]
        train_chunks.append(chunk[chunk["t_dat"] < TRAIN_END])
        val_chunks.append(
            chunk[(chunk["t_dat"] >= TRAIN_END) & (chunk["t_dat"] < VAL_END)]
        )
        test_chunks.append(
            chunk[(chunk["t_dat"] >= VAL_END) & (chunk["t_dat"] < TEST_END)]
        )
        if i % 5 == 0:
            print(f"  Processed {(i+1)*chunk_size/1_000_000:.0f}M+ rows...")

    print("\nConcatenating and saving splits...")

    print("\n[Train]")
    train = pd.concat(train_chunks, ignore_index=True)
    del train_chunks
    gc.collect()
    process_and_save(train, "train")
    del train
    gc.collect()

    print("\n[Val]")
    val = pd.concat(val_chunks, ignore_index=True)
    del val_chunks
    gc.collect()
    process_and_save(val, "val")
    del val
    gc.collect()

    print("\n[Test]")
    test = pd.concat(test_chunks, ignore_index=True)
    del test_chunks
    gc.collect()
    process_and_save(test, "test")
    del test
    gc.collect()

    print("\nDone. Files in data/processed/:")
    for f in sorted(PROCESSED_DIR.glob("*.parquet")):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")