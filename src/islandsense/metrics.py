"""Model evaluation metrics for IslandSense.

Includes Expected Calibration Error (ECE) for assessing probability calibration.
"""

import numpy as np


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 5
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE measures how well predicted probabilities match actual outcomes.
    Lower is better (0 = perfect calibration).

    Algorithm:
    1. Bin predictions into n_bins buckets
    2. For each bin, compute |mean_predicted_prob - mean_actual_outcome|
    3. Weight by bin size and average

    Args:
        y_true: True binary labels (0/1)
        y_prob: Predicted probabilities [0, 1]
        n_bins: Number of bins for calibration curve (default 5)

    Returns:
        ECE score (0 = perfect calibration)
    """
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])  # Bin assignment for each sample

    ece = 0.0
    total_count = len(y_true)

    for bin_idx in range(n_bins):
        # Find samples in this bin
        in_bin = bin_indices == bin_idx
        bin_count = in_bin.sum()

        if bin_count == 0:
            continue

        # Mean predicted probability in this bin
        mean_predicted = y_prob[in_bin].mean()

        # Mean actual outcome in this bin (fraction of positives)
        mean_actual = y_true[in_bin].mean()

        # Weighted absolute difference
        ece += (bin_count / total_count) * abs(mean_predicted - mean_actual)

    return ece


def print_calibration_bins(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 5
) -> None:
    """Print calibration table for inspection.

    Args:
        y_true: True binary labels (0/1)
        y_prob: Predicted probabilities [0, 1]
        n_bins: Number of bins (default 5)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    print("  Calibration Bins:")
    print("  " + "=" * 60)
    print(
        f"  {'Bin':>8} {'Range':>15} {'Count':>8} {'Pred':>8} {'Actual':>8} {'Error':>8}"
    )
    print("  " + "-" * 60)

    for bin_idx in range(n_bins):
        in_bin = bin_indices == bin_idx
        bin_count = in_bin.sum()

        if bin_count == 0:
            continue

        bin_min = bin_edges[bin_idx]
        bin_max = bin_edges[bin_idx + 1]
        mean_predicted = y_prob[in_bin].mean()
        mean_actual = y_true[in_bin].mean()
        error = abs(mean_predicted - mean_actual)

        print(
            f"  {bin_idx + 1:>8} [{bin_min:.2f}, {bin_max:.2f}] {bin_count:>8} "
            f"{mean_predicted:>8.3f} {mean_actual:>8.3f} {error:>8.3f}"
        )

    print("  " + "=" * 60)
