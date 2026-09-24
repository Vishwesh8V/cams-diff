import argparse
import csv
import json
import os
import sys


try:
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog('rdApp.*')
    HAVE_RDKIT = True

except ImportError:
    HAVE_RDKIT = False


def canonicalize(smi):
    """RDKit canonicalization, Kekulized. Returns None if RDKit rejects the SMILES.

    Keep this identical to the retrosynthesis converter so the forward dataset
    uses exactly the same SMILES representation and tokenizer vocabulary.
    """
    if not HAVE_RDKIT:
        return smi

    mol = Chem.MolFromSmiles(smi)

    if mol is None:
        return None

    try:
        Chem.Kekulize(
            mol,
            clearAromaticFlags=True
        )
    except Chem.KekulizeException:
        return None

    return Chem.MolToSmiles(
        mol,
        kekuleSmiles=True
    )


def load_molinstructions(path):


    try:
        import selfies as sf

    except ImportError:
        sys.exit(
            "Mol-Instructions is SELFIES-encoded; "
            "install with: pip install selfies"
        )

    with open(
        path,
        'r',
        encoding='utf-8'
    ) as f:
        data = json.load(f)

    out = []

    n_bad = 0

    for info in data:

        try:
            # Forward direction:
            #
            # input  -> reactants
            # output -> product

            reactant_smiles = sf.decoder(
                info['input'],
                compatible=True
            )

            product_smiles = sf.decoder(
                info['output'],
                compatible=True
            )

        except Exception:

            n_bad += 1
            continue

        split = (
            info
            .get('metadata', {})
            .get('split', 'train')
        )

        split = {
            'valid': 'validation'
        }.get(
            split,
            split
        )

        out.append(
            (
                split,
                product_smiles,
                reactant_smiles
            )
        )

    if n_bad:

        print(
            f"[build_forward_dataset] "
            f"{n_bad} examples failed SELFIES decoding "
            f"and were dropped"
        )

    return out


def load_uspto50k(
    path,
    rxn_col=None,
    split_col=None,
    delimiter=None
):
    """
    Generic reaction-SMILES file
    -> list of (split, reactant_smiles, product_smiles).

    Auto-detects the common USPTO-50k Kaggle layout
    (id,class,reactants>reagents>production,split)
    if columns aren't specified.
    """

    with open(
        path,
        'r',
        encoding='utf-8'
    ) as f:

        sniff = f.read(4096)

        f.seek(0)

        if delimiter is None:

            delimiter = (
                ','
                if sniff.count(',') > sniff.count('\t')
                else '\t'
            )

        reader = csv.reader(
            f,
            delimiter=delimiter
        )

        rows = list(reader)

    if not rows:
        return []

    header = rows[0]

    has_header = (
        any(
            '>' not in c
            and not c.replace('.', '').isdigit()
            for c in header
        )
        and
        not any(
            '>' in c
            for c in header
        )
    )

    if has_header:

        header_lower = [
            h.strip().lower()
            for h in header
        ]

        if rxn_col is None:

            for i, h in enumerate(header_lower):

                if (
                    'reagent' in h
                    or 'reaction' in h
                    or 'smiles' in h
                    or 'production' in h
                ):

                    rxn_col = i
                    break

        if (
            split_col is None
            and 'split' in header_lower
        ):

            split_col = header_lower.index(
                'split'
            )

        data_rows = rows[1:]

    else:

        data_rows = rows

        if rxn_col is None:
            rxn_col = 0

    if rxn_col is None:

        sys.exit(
            "Could not auto-detect the reaction-SMILES "
            "column. Pass --rxn_col explicitly "
            "(0-indexed) after checking a few lines "
            "of your file."
        )

    out = []

    n_bad = 0

    for row in data_rows:

        if rxn_col >= len(row):
            continue

        rxn = row[rxn_col].strip()

        parts = rxn.split('>')

        if len(parts) == 3:

            reactants, _reagents, product = parts

        elif len(parts) == 2:

            reactants, product = parts

        else:

            n_bad += 1
            continue

        if not reactants or not product:

            n_bad += 1
            continue

        split = (
            row[split_col].strip()
            if (
                split_col is not None
                and split_col < len(row)
            )
            else 'train'
        )

        out.append(
            (
                split,
                product,
                reactants
            )
        )

    if n_bad:

        print(
            f"[build_forward_dataset] "
            f"{n_bad} rows had an unparseable "
            f"reaction SMILES and were dropped"
        )

    return out


def write_splits(
    examples,
    out_dir
):
    """
    Write:

        train.txt
        val.txt
        test.txt

    with:

        cid    smiles    desc

    where:

        smiles = product
        desc   = reactants
    """

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    by_split = {}

    for split, product, reactants in examples:

        by_split.setdefault(
            split,
            []
        ).append(
            (
                product,
                reactants
            )
        )

    split_to_filename = {
        'train': 'train.txt',
        'validation': 'validation.txt',
        'test': 'test.txt'
    }

    unknown = (
        set(by_split)
        -
        set(split_to_filename)
    )

    if unknown:

        print(
            f"[build_forward_dataset] "
            f"Unrecognized split label(s) {unknown} "
            f"-- writing them into train.txt"
        )

        by_split.setdefault(
            'train',
            []
        )

        for u in unknown:

            by_split['train'].extend(
                by_split.pop(u)
            )

    counts = {}

    for split, fname in split_to_filename.items():

        rows = by_split.get(
            split,
            []
        )

        path = os.path.join(
            out_dir,
            fname
        )

        n_written = 0
        n_skipped = 0

        with open(
            path,
            'w',
            encoding='utf-8'
        ) as f:

            # Same format as build_retro_dataset.py.
            #
            # ChEBIdataset.get_ori_data() skips line 0.

            f.write(
                'cid\tsmiles\tdesc\n'
            )

            for cid, (
                product,
                reactants
            ) in enumerate(rows):

                # Forward:
                #
                # product -> diffusion target
                # reactants -> conditioning

                can_product = canonicalize(
                    product
                )

                can_reactants_parts = [
                    canonicalize(p)
                    for p in reactants.split('.')
                ]

                if (
                    can_product is None
                    or any(
                        p is None
                        for p in can_reactants_parts
                    )
                ):

                    n_skipped += 1
                    continue

                can_reactants = '.'.join(
                    can_reactants_parts
                )

                # IMPORTANT:
                #
                # Same physical file format as RETRO:
                #
                #   cid | smiles | desc
                #
                # But FORWARD reverses what occupies
                # those columns:
                #
                #   smiles = product
                #   desc   = reactants

                f.write(
                    f"{cid}\t"
                    f"{can_product}\t"
                    f"{can_reactants}\n"
                )

                n_written += 1

        counts[fname] = (
            n_written,
            n_skipped
        )

    return counts


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        '--input',
        required=True,
        help='Path to the raw forward reaction data file'
    )

    ap.add_argument(
        '--format',
        choices=[
            'molinstructions',
            'uspto50k'
        ],
        required=True
    )

    ap.add_argument(
        '--out_dir',
        default='../../datasets/FORWARD'
    )

    ap.add_argument(
        '--rxn_col',
        type=int,
        default=None,
        help='(uspto50k) 0-indexed column with the reaction SMILES'
    )

    ap.add_argument(
        '--split_col',
        type=int,
        default=None,
        help='(uspto50k) 0-indexed column with the split label'
    )

    ap.add_argument(
        '--delimiter',
        default=None,
        help='(uspto50k) override delimiter auto-detection'
    )

    args = ap.parse_args()

    if not HAVE_RDKIT:

        print(
            "[build_forward_dataset] WARNING: "
            "rdkit not importable -- skipping "
            "canonicalization/validation. "
            "ChEBIdataset does not validate SMILES itself, "
            "so garbage input will silently reach training."
        )

    if args.format == 'molinstructions':

        examples = load_molinstructions(
            args.input
        )

    else:

        examples = load_uspto50k(
            args.input,
            args.rxn_col,
            args.split_col,
            args.delimiter
        )

    if not examples:

        sys.exit(
            "No examples parsed -- check --format, "
            "input file, and reaction column settings."
        )

    counts = write_splits(
        examples,
        args.out_dir
    )

    print(
        f"\nWrote dataset to {args.out_dir}:"
    )

    for fname, (
        written,
        skipped
    ) in counts.items():

        print(
            f"  {fname}: {written} examples written, "
            f"{skipped} dropped "
            f"(failed RDKit sanitization)"
        )

    print(
        "\nForward role mapping:"
    )

    print(
        "  smiles = product "
        "(diffusion target)"
    )

    print(
        "  desc   = reactants "
        "(cross-attention conditioning)"
    )

    print(
        "\nNext: run build_retro_vocab.py against "
        "this directory to generate generate_vocab.txt, "
        "then process_text.py --dataset_dir <out_dir> "
        "for each split."
    )


if __name__ == '__main__':
    main()