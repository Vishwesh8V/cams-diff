import argparse
import numpy as np


def build_step_matrix(steps, seq_len, num_timesteps):
    steps = np.asarray(steps, dtype=np.int64)
    K = steps.shape[0]

    if K < 2:
        raise ValueError(f"Need at least 2 steps (got {K}).")
    if not np.all(steps[:-1] > steps[1:]):
        raise ValueError(
            f"--steps must be strictly decreasing; got {steps.tolist()}"
        )
    if steps[0] != num_timesteps - 1:
        raise ValueError(
            f"First step must be num_timesteps - 1 = {num_timesteps - 1} "
            f"(the noisiest raw timestep); got {steps[0]}. "
            f"TokenAdaptiveDPMSolver's row-0 model call is meant to read "
            f"the raw sampling prior, which corresponds to this index."
        )
    if steps[-1] != 0:
        raise ValueError(
            f"Last step must be 0 (the cleanest raw timestep); got {steps[-1]}."
        )
    if steps.min() < 0 or steps.max() >= num_timesteps:
        raise ValueError(
            f"All steps must be in [0, {num_timesteps - 1}]; got "
            f"min={steps.min()}, max={steps.max()}."
        )

    # Broadcast the SAME list to every position -- this is what makes it
    # "uniform trajectory," the same way Stage 4's fallback for inactive
    # positions works, just applied to every position here.
    J = np.repeat(steps[:, None], seq_len, axis=1)
    assert J.shape == (K, seq_len)
    return J


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", type=int, nargs="+", required=True,
        help="Strictly-decreasing raw timestep indices to visit, e.g. "
             "1999 1500 1000 500 100 0. First must be num_timesteps-1, "
             "last must be 0."
    )
    parser.add_argument(
        "--seq_len", type=int, required=True,
        help="tokenizer.max_len for the checkpoint you're sampling from "
             "(L in the [K, L] step matrix) -- must match exactly or "
             "load_step_matrix() will reject the file."
    )
    parser.add_argument(
        "--num_timesteps", type=int, default=2000,
        help="Diffusion steps the checkpoint was trained with (T). "
             "Default 2000 matches this project's training scripts."
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    J = build_step_matrix(args.steps, args.seq_len, args.num_timesteps)

    np.save(args.output, J)
    print(f"Saved step matrix: {args.output}")
    print(f"Shape: {J.shape}  (K={J.shape[0]} steps, L={J.shape[1]} positions)")
    print(f"Steps visited (same for every position): {args.steps}")
    print()
    print("Run sampling with:")
    print(f"  --step_matrix_path {args.output}")
    print(f"  --token_adaptive_steps {J.shape[0]}")
    print("(and do not also pass --use_dpm_solver -- they're mutually "
          "exclusive samplers in text_sample.py)")


if __name__ == "__main__":
    main()