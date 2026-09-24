import os
import re
import argparse

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import GraphDescriptors
from rdkit import DataStructs


DEFAULT_TEST_FILE = (
    "datasets/SMILES/test.txt"
)

DEFAULT_SEED_FILE = (
    ""
    ""
)

DEFAULT_OUTPUT_DIR = (
    "scalability_outputs_1seed"
)



def load_test_data(filepath):

    records = []

    with open(
        filepath,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(f):

            line = line.rstrip("\n")

            if not line.strip():
                continue

            parts = line.split("\t")

            if len(parts) < 3:
                print(
                    f"WARNING: Could not parse line "
                    f"{line_number}: {line[:150]}"
                )
                continue

            cid = parts[0].strip()
            smiles = parts[1].strip()
            description = "\t".join(
                parts[2:]
            ).strip()

            # Skip header
            if cid.lower() == "cid":
                continue

            records.append({
                "test_index": len(records),
                "file_line": line_number,
                "cid": cid,
                "ground_truth": smiles,
                "description": description,
            })

    print(
        f"\nLoaded {len(records)} test examples."
    )

    return records


def load_generation_file(filepath):

    generated = []

    with open(
        filepath,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(f):

            line = line.strip()

            if not line:
                continue

            # Your generation format:
            #
            # generated_smiles || ground_truth_smiles

            if "||" not in line:

                print(
                    f"WARNING: No || delimiter at "
                    f"generation line {line_number}"
                )

                generated.append("")
                continue

            generated_smiles = (
                line
                .split("||", 1)[0]
                .strip()
            )

            # Remove possible special tokens
            special_tokens = [
                "[PAD]",
                "[SOS]",
                "[EOS]",
                "[X]",
                "[XPara]",
                "[XRing]",
            ]

            for token in special_tokens:
                generated_smiles = (
                    generated_smiles
                    .replace(token, "")
                )

            generated_smiles = (
                generated_smiles.strip()
            )

            generated.append(
                generated_smiles
            )

    print(
        f"Loaded {len(generated)} generated "
        f"molecules."
    )

    return generated


def word_length(text):

    return len(
        text.split()
    )


def load_roberta_tokenizer():
    try:

        from transformers import (
            AutoTokenizer
        )

        tokenizer = AutoTokenizer.from_pretrained(
            "roberta-base"
        )

        print(
            "\nLoaded RoBERTa tokenizer."
        )

        return tokenizer

    except Exception as e:

        print(
            "\nWARNING: Could not load "
            "RoBERTa tokenizer."
        )

        print(
            f"Reason: {e}"
        )

        return None


def tokenizer_length(
    text,
    tokenizer
):

    if tokenizer is None:
        return np.nan

    try:

        tokens = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False
        )

        return len(
            tokens["input_ids"]
        )

    except Exception:

        return np.nan



def length_bucket(length):

    if np.isnan(length):
        return None

    if length <= 64:
        return "1-64"

    elif length <= 96:
        return "65-96"

    elif length <= 128:
        return "97-128"

    elif length <= 160:
        return "129-160"

    else:
        return ">160"



def calculate_bertzct(smiles):

    try:

        mol = Chem.MolFromSmiles(
            smiles
        )

        if mol is None:
            return np.nan

        return float(
            GraphDescriptors.BertzCT(mol)
        )

    except Exception:

        return np.nan


def bertz_bucket(value):

    if np.isnan(value):
        return None

    if value < 100:
        return "<100"

    elif value < 300:
        return "100-300"

    elif value < 800:
        return "300-800"

    else:
        return ">800"


def calculate_maccs(
    generated,
    ground_truth
):

    try:

        gen_mol = Chem.MolFromSmiles(
            generated
        )

        gt_mol = Chem.MolFromSmiles(
            ground_truth
        )

        if (
            gen_mol is None
            or gt_mol is None
        ):
            return np.nan, False

        gen_fp = (
            MACCSkeys.GenMACCSKeys(
                gen_mol
            )
        )

        gt_fp = (
            MACCSkeys.GenMACCSKeys(
                gt_mol
            )
        )

        score = (
            DataStructs.FingerprintSimilarity(
                gen_fp,
                gt_fp
            )
        )

        return float(score), True

    except Exception:

        return np.nan, False



def load_generation_file_camsdiff(filepath):

    generated = []

    with open(
        filepath,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.rstrip("\n")

            if not line.strip():
                generated.append("")
                continue

            generated_smiles = (
                line.split(" ")[0]
            )

            generated.append(
                generated_smiles.strip()
            )

    print(
        f"Loaded {len(generated)} generated "
        f"molecules (cams plain format)."
    )

    return generated


# ============================================================
# CANONICALIZATION / EXACT MATCH
# ============================================================

def canonicalize(smiles):

    try:

        mol = Chem.MolFromSmiles(
            smiles
        )

        if mol is None:
            return None

        return Chem.MolToSmiles(
            mol,
            canonical=True
        )

    except Exception:

        return None


def calculate_exact_match(
    generated,
    ground_truth
):
    """
    Canonical-SMILES exact match, consistent with the
    convention used in pcdes_zero_shot.py.

    Returns True only if BOTH strings parse and their
    canonical forms are identical. This is independent of
    calculate_maccs: a molecule can be canonically inequal
    yet still receive a defined (low) MACCS score.
    """

    gen_can = canonicalize(generated)
    gt_can = canonicalize(ground_truth)

    return (
        gen_can is not None
        and gt_can is not None
        and gen_can == gt_can
    )


# ============================================================
# PROCESS ONE SEED
# ============================================================

def process_seed(
    test_records,
    generated,
    tokenizer
):

    n = min(
        len(test_records),
        len(generated)
    )

    if (
        len(test_records)
        != len(generated)
    ):

        print(
            "\nWARNING:"
        )

        print(
            f"Test examples     : "
            f"{len(test_records)}"
        )

        print(
            f"Generated examples: "
            f"{len(generated)}"
        )

        print(
            f"Using first {n} examples."
        )

    results = []

    for i in range(n):

        record = test_records[i]

        gt = record[
            "ground_truth"
        ]

        gen = generated[i]

        # ----------------------------------------------------
        # Instruction length
        # ----------------------------------------------------

        words = word_length(
            record["description"]
        )

        tokens = tokenizer_length(
            record["description"],
            tokenizer
        )

        word_bucket = length_bucket(
            words
        )

        token_bucket = length_bucket(
            tokens
        )

        # ----------------------------------------------------
        # BertzCT
        # ----------------------------------------------------

        bertz = calculate_bertzct(
            gt
        )

        bertz_group = bertz_bucket(
            bertz
        )

        # ----------------------------------------------------
        # MACCS
        # ----------------------------------------------------

        maccs, both_parsed = calculate_maccs(
            gen,
            gt
        )

        # ----------------------------------------------------
        # Validity (of the GENERATED molecule only — decoupled
        # from whether the reference happens to parse, which
        # `both_parsed` above conflates)
        # ----------------------------------------------------

        gen_valid = (
            Chem.MolFromSmiles(gen) is not None
            if gen else False
        )

        # ----------------------------------------------------
        # Exact match (canonical SMILES)
        # ----------------------------------------------------

        exact = calculate_exact_match(
            gen,
            gt
        )

        results.append({

            "test_index":
                record["test_index"],

            "cid":
                record["cid"],

            "description":
                record["description"],

            "ground_truth":
                gt,

            "generated":
                gen,

            "word_length":
                words,

            "word_bucket":
                word_bucket,

            "token_length":
                tokens,

            "token_bucket":
                token_bucket,

            "bertzct":
                bertz,

            "bertz_bucket":
                bertz_group,

            "maccs":
                maccs,

            "valid":
                both_parsed,   # kept for backward compatibility

            "gen_valid":
                gen_valid,

            "exact_match":
                exact,
        })

    return pd.DataFrame(
        results
    )


# ============================================================
# BUCKET SUMMARY
# ============================================================

def summarize(
    df,
    bucket_column,
    bucket_order
):

    rows = []

    for bucket in bucket_order:

        subset = df[
            df[bucket_column] == bucket
        ]

        all_count = len(
            subset
        )

        valid = subset[
            subset["maccs"].notna()
        ]

        valid_count = len(
            valid
        )

        if valid_count > 0:

            mean_maccs = (
                valid["maccs"].mean()
            )

            std_maccs = (
                valid["maccs"].std()
                if valid_count > 1
                else 0.0
            )

        else:

            mean_maccs = np.nan
            std_maccs = np.nan

        # ------------------------------------------------
        # Validity: fraction of GENERATED molecules that
        # parse, out of all rows in the bucket. This is
        # independent of whether MACCS is defined for the
        # row (that also requires the reference to parse).
        # ------------------------------------------------

        gen_valid_count = int(
            subset["gen_valid"].sum()
        )

        validity_rate = (
            gen_valid_count / all_count
            if all_count > 0
            else np.nan
        )

        # ------------------------------------------------
        # Exact match: canonical-SMILES equality, out of
        # all rows in the bucket.
        # ------------------------------------------------

        exact_count = int(
            subset["exact_match"].sum()
        )

        exact_rate = (
            exact_count / all_count
            if all_count > 0
            else np.nan
        )

        rows.append({

            "bucket":
                bucket,

            "MACCS_FTS":
                mean_maccs,

            "STD":
                std_maccs,

            "N":
                all_count,

            "N_valid":
                valid_count,

            "validity":
                (
                    valid_count / all_count
                    if all_count > 0
                    else np.nan
                ),

            "N_gen_valid":
                gen_valid_count,

            "Validity":
                validity_rate,

            "N_exact":
                exact_count,

            "Exact":
                exact_rate,
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# PRINT DISTRIBUTION
# ============================================================

def print_distribution(df):

    print("\n")
    print("=" * 100)
    print("DATASET COMPLEXITY DISTRIBUTION")
    print("=" * 100)

    # --------------------------------------------------------
    # Word lengths
    # --------------------------------------------------------

    print("\nWORD LENGTH")
    print("-" * 100)

    print(
        df[
            "word_length"
        ].describe()
    )

    print("\nWord-length buckets:")

    print(
        df[
            "word_bucket"
        ]
        .value_counts(
            sort=False
        )
    )

    # --------------------------------------------------------
    # Token lengths
    # --------------------------------------------------------

    print("\nTOKEN LENGTH")
    print("-" * 100)

    if df["token_length"].notna().any():

        print(
            df[
                "token_length"
            ].describe()
        )

        print(
            "\nToken-length buckets:"
        )

        order = [
            "1-64",
            "65-96",
            "97-128",
            "129-160",
            ">160",
        ]

        counts = (
            df["token_bucket"]
            .value_counts(
                sort=False
            )
            .reindex(order)
            .fillna(0)
            .astype(int)
        )

        print(counts)

    else:

        print(
            "Tokenizer lengths unavailable."
        )

    # --------------------------------------------------------
    # BertzCT
    # --------------------------------------------------------

    print("\nBERTZCT")
    print("-" * 100)

    valid_bertz = df[
        df["bertzct"].notna()
    ]["bertzct"]

    if len(valid_bertz) > 0:

        print(
            valid_bertz.describe()
        )

        print(
            "\nBertzCT buckets:"
        )

        order = [
            "<100",
            "100-300",
            "300-800",
            ">800",
        ]

        counts = (
            df["bertz_bucket"]
            .value_counts(
                sort=False
            )
            .reindex(order)
            .fillna(0)
            .astype(int)
        )

        print(counts)

    else:

        print(
            "NO VALID BERTZCT VALUES!"
        )

    print("=" * 100)


# ============================================================
# PRINT TABLE
# ============================================================

def print_table(
    title,
    table,
    model_name="",
    metric_column="MACCS_FTS"
):

    print("\n")
    print("=" * 100)

    print(title)

    print("=" * 100)

    print(
        f"{'Model':<18}",
        end=""
    )

    for bucket in table["bucket"]:

        print(
            f"{bucket:>12}",
            end=""
        )

    print()

    print("-" * 100)

    print(
        f"{model_name:<18}",
        end=""
    )

    for value in table[
        metric_column
    ]:

        if pd.isna(value):

            text = "N/A"

        else:

            text = f"{value:.3f}"

        print(
            f"{text:>12}",
            end=""
        )

    print()

    print("=" * 100)


# ============================================================
# PRINT DETAILED TABLE
# ============================================================

def print_detailed_table(
    title,
    table
):

    print("\n")
    print(title)

    print("-" * 100)

    print(
        f"{'Bucket':<15}"
        f"{'MACCS':>12}"
        f"{'STD':>12}"
        f"{'N':>10}"
        f"{'Valid':>10}"
        f"{'Validity':>12}"
        f"{'Exact':>10}"
    )

    print("-" * 100)

    for _, row in table.iterrows():

        if pd.isna(
            row["MACCS_FTS"]
        ):

            score = "N/A"
            std = "N/A"

        else:

            score = (
                f"{row['MACCS_FTS']:.4f}"
            )

            std = (
                f"{row['STD']:.4f}"
            )

        validity = (
            f"{row['Validity']:.3f}"
            if not pd.isna(
                row["Validity"]
            )
            else "N/A"
        )

        exact = (
            f"{row['Exact']:.3f}"
            if not pd.isna(
                row["Exact"]
            )
            else "N/A"
        )

        print(
            f"{row['bucket']:<15}"
            f"{score:>12}"
            f"{std:>12}"
            f"{int(row['N']):>10}"
            f"{int(row['N_gen_valid']):>10}"
            f"{validity:>12}"
            f"{exact:>10}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_file",
        default=DEFAULT_TEST_FILE
    )

    parser.add_argument(
        "--seed_file",
        default=DEFAULT_SEED_FILE
    )

    parser.add_argument(
        "--gen_format",
        choices=["cams", "utgdiff"],
        default="cams",
        help=(
            "cams: lines are 'generated || ground_truth' "
            "(ground truth column is ignored either way; "
            "always taken from --test_file). "
            "utgdiff: lines are a bare generated SMILES, "
            "aligned by row order with --test_file, as in "
            "UTGDiff's own generation_results/*.txt."
        )
    )

    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR
    )

    args = parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    print("=" * 100)
    print("")
    print("=" * 100)

    print(
        f"\nTest file:\n{args.test_file}"
    )

    print(
        f"\nGeneration file:\n{args.seed_file}"
    )

    # ========================================================
    # LOAD
    # ========================================================

    test_records = load_test_data(
        args.test_file
    )

    generated = (
        load_generation_file_camsdiff(
            args.seed_file
        )
        if args.gen_format == "utgdiff"
        else load_generation_file(
            args.seed_file
        )
    )

    tokenizer = load_roberta_tokenizer()

    # ========================================================
    # PROCESS
    # ========================================================

    df = process_seed(
        test_records,
        generated,
        tokenizer
    )

    # ========================================================
    # SAVE RAW DATA
    # ========================================================

    raw_path = os.path.join(
        args.output_dir,
        "sample_level_results.csv"
    )

    df.to_csv(
        raw_path,
        index=False
    )

    print(
        f"\nSaved sample-level results:\n"
        f"{raw_path}"
    )

    # ========================================================
    # DISTRIBUTION DIAGNOSTICS
    # ========================================================

    print_distribution(
        df
    )

    # ========================================================
    # LENGTH TABLES
    # ========================================================

    length_order = [
        "1-64",
        "65-96",
        "97-128",
        "129-160",
        ">160",
    ]

    word_table = summarize(
        df,
        "word_bucket",
        length_order
    )

    token_table = summarize(
        df,
        "token_bucket",
        length_order
    )

    # ========================================================
    # BERTZ TABLE
    # ========================================================

    bertz_order = [
        "<100",
        "100-300",
        "300-800",
        ">800",
    ]

    bertz_table = summarize(
        df,
        "bertz_bucket",
        bertz_order
    )

    # ========================================================
    # PRINT PAPER-STYLE TABLES
    # ========================================================

    for metric_col, metric_label in [
        ("MACCS_FTS", "MACCS"),
        ("Validity", "VALIDITY"),
        ("Exact", "EXACT MATCH"),
    ]:

        print_table(
            f"TABLE 4-STYLE: INSTRUCTION LENGTH "
            f"(WORD COUNT) — {metric_label}",
            word_table,
            metric_column=metric_col
        )

        print_table(
            f"TABLE 4-STYLE: INSTRUCTION LENGTH "
            f"(TOKEN COUNT) — {metric_label}",
            token_table,
            metric_column=metric_col
        )

        print_table(
            f"TABLE 4-STYLE: MOLECULE COMPLEXITY "
            f"(BERTZCT) — {metric_label}",
            bertz_table,
            metric_column=metric_col
        )

    # ========================================================
    # PRINT DETAILED TABLES
    # ========================================================

    print_detailed_table(
        "WORD LENGTH — DETAILED",
        word_table
    )

    print_detailed_table(
        "TOKEN LENGTH — DETAILED",
        token_table
    )

    print_detailed_table(
        "BERTZCT — DETAILED",
        bertz_table
    )

    # ========================================================
    # SAVE TABLES
    # ========================================================

    word_path = os.path.join(
        args.output_dir,
        "instruction_length_words.csv"
    )

    token_path = os.path.join(
        args.output_dir,
        "instruction_length_tokens.csv"
    )

    bertz_path = os.path.join(
        args.output_dir,
        "bertzct.csv"
    )

    word_table.to_csv(
        word_path,
        index=False
    )

    token_table.to_csv(
        token_path,
        index=False
    )

    bertz_table.to_csv(
        bertz_path,
        index=False
    )

    # ========================================================
    # SAVE FINAL TEXT TABLE
    # ========================================================

    final_path = os.path.join(
        args.output_dir,
        ""
    )

    with open(
        final_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            ""
        )

        f.write(
            "Metrics: MACCS Fingerprint Tanimoto "
            "Similarity, Validity, Exact Match "
            "(canonical SMILES)\n\n"
        )

        def _write_block(
            section_title,
            bucket_order,
            table
        ):

            f.write(
                f"{section_title}\n"
            )

            f.write(
                "Model\t"
                + "\t".join(
                    bucket_order
                )
                + "\n"
            )

            for metric_col, metric_label in [
                ("MACCS_FTS", "MACCS"),
                ("Validity", "Validity"),
                ("Exact", "Exact"),
            ]:

                f.write(
                    f""
                    + "\t".join(
                        "N/A"
                        if pd.isna(v)
                        else f"{v:.3f}"
                        for v in table[metric_col]
                    )
                    + "\n"
                )

            f.write("\n")

        _write_block(
            "Instruction Length (Word Count)",
            length_order,
            word_table
        )

        _write_block(
            "Instruction Length (Token Count)",
            length_order,
            token_table
        )

        _write_block(
            "BertzCT",
            bertz_order,
            bertz_table
        )

    print(
        f"\nFinal table saved to:\n"
        f"{final_path}"
    )

    print("\n")
    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()