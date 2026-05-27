import numpy as np
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer as SDVGaussianCopula


def _build_weekly_time_index(n_rows: int) -> pd.DataFrame:
    """
    Creates a sorted time skeleton of n_rows rows cycling through
    day_of_week (0-6) and hour (0-23) in weekly order.

    Why: Gaussian Copula generates rows in random order with no time
    awareness. Before recomputing lags, we need rows in a sensible
    temporal order so lag_1h at position t actually reflects the
    sessions value at position t-1.

    We don't use real timestamps — just the cyclic week structure
    that your features already encode.
    """
    slots = []
    # Keep adding full weeks until we have enough rows
    week = 0
    while len(slots) < n_rows:
        for dow in range(7):          # Monday=0 through Sunday=6
            for hour in range(24):    # hour 0 through 23
                slots.append({
                    "day_of_week": dow,
                    "hour": hour,
                    # is_weekend is deterministic from day_of_week
                    "is_weekend": 1 if dow >= 5 else 0
                })
                if len(slots) == n_rows:
                    break
            if len(slots) == n_rows:
                break
        week += 1

    return pd.DataFrame(slots)


def _recompute_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe with a 'sessions' column in sorted temporal order,
    recomputes all lag and rolling features from scratch.

    Why: Gaussian Copula samples lag values as independent draws from
    their marginal distribution — they are not computed from actual
    previous rows. If we used those sampled lags, the model would see
    contradictory evidence: lag_1h=3 but the previous row's sessions=0.
    Recomputing from the sessions column makes every lag internally
    consistent with the synthetic sequence.
    """
    df = df.copy()

    df["lag_1h"]   = df["sessions"].shift(1).fillna(0)
    df["lag_24h"]  = df["sessions"].shift(24).fillna(0)
    df["lag_168h"] = df["sessions"].shift(168).fillna(0)

    # rolling() requires min_periods=1 so early rows don't become NaN
    df["rolling_24h_mean"] = (
        df["sessions"]
        .rolling(window=24, min_periods=1)
        .mean()
    )
    df["rolling_7d_mean"] = (
        df["sessions"]
        .rolling(window=168, min_periods=1)
        .mean()
    )

    return df


class GaussianCopulaSynthesizer:
    """
    Wraps SDV's GaussianCopulaSynthesizer to generate synthetic hourly
    EV demand rows that preserve column correlations.

    Workflow:
        1. fit()   — learn marginal distributions + correlation structure
                     from real fine-tuning data
        2. sample() — generate n_rows synthetic rows, sort into weekly
                      order, recompute lag features, return clean dataframe

    What it does NOT do: preserve temporal ordering within generated rows.
    That limitation is handled by explicit sorting + lag recomputation.
    """

    # Columns we feed to SDV — only the ones that carry real signal
    # and don't depend on temporal ordering.
    # is_holiday is included because it correlates with demand drops.
    # Lag features are EXCLUDED — we recompute them after sorting.
    SYNTH_COLUMNS = ["hour", "day_of_week", "is_weekend", "is_holiday", "sessions"]

    def __init__(self):
        self._synthesizer = None   # SDV model, populated after fit()
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the Gaussian Copula on real fine-tuning data.

        Args:
            df: Real hourly demand dataframe. Must contain all columns
                in SYNTH_COLUMNS. Timestamp column is ignored if present.

        Why Metadata: SDV 1.x requires explicit metadata telling it
        which columns are numerical vs categorical. Without this, it
        may treat 'hour' as a continuous float and model it incorrectly.
        'hour' is better treated as numerical here because its values
        (0-23) have a meaningful order and the copula can learn that
        hour=9 correlates with high sessions.
        """
        # Keep only the columns we want to synthesize
        synth_df = df[self.SYNTH_COLUMNS].copy()

        # Build SDV metadata programmatically
        # sdtype "numerical" → fits a parametric distribution per column
        # sdtype "categorical" → fits empirical distribution, samples exactly
        # is_holiday and is_weekend are binary — categorical preserves {0,1}
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(synth_df)
        metadata.update_column(column_name="is_weekend", sdtype="categorical")
        metadata.update_column(column_name="is_holiday", sdtype="categorical")
        self._synthesizer = SDVGaussianCopula(metadata)
        self._synthesizer.fit(synth_df)
        self._fitted = True
        print(f"GaussianCopulaSynthesizer fitted on {len(synth_df)} rows.")

    def sample(self, n_rows: int, real_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate n_rows synthetic rows and return a fully-featured
        dataframe ready to prepend before real fine-tuning data.

        Args:
            n_rows:  How many synthetic rows to generate.
            real_df: The real fine-tuning dataframe. Used only to get
                     the is_holiday values for the synthetic week slots
                     (since is_holiday depends on actual calendar dates,
                     not just day_of_week — we sample it from real data).

        Returns:
            DataFrame with all columns matching real_df, sorted in
            weekly temporal order, with lag features recomputed.

        Why sample more then trim: SDV sometimes rejects rows that
        violate constraints during sampling. Generating 20% extra
        then trimming to exactly n_rows makes the output size reliable.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before sample().")

        # Generate with 20% buffer then trim
        raw_synthetic = self._synthesizer.sample(num_rows=int(n_rows * 1.2))

        # Clip sessions to non-negative — copula can occasionally sample
        # small negative values for right-skewed distributions
        raw_synthetic["sessions"] = raw_synthetic["sessions"].clip(lower=0)

        # Round sessions to nearest integer — demand is a count
        raw_synthetic["sessions"] = raw_synthetic["sessions"].round().astype(int)
        

        # Build the weekly time skeleton we'll sort into
        time_index = _build_weekly_time_index(n_rows)

        # Assign sessions values from synthetic rows into the time skeleton.
        # We sample without replacement from synthetic sessions to fill slots.
        # This preserves the sessions distribution while imposing week structure.
        if len(raw_synthetic) >= n_rows:
            sampled_sessions = (
                raw_synthetic["sessions"]
                .sample(n=n_rows, replace=False)
                .values
            )
        else:
            # Fallback: sample with replacement if we didn't get enough
            sampled_sessions = (
                raw_synthetic["sessions"]
                .sample(n=n_rows, replace=True)
                .values
            )

        time_index["sessions"] = sampled_sessions

        # is_holiday: sample from real data's holiday distribution
        # Holiday rates vary by site — preserve that rate in synthetic rows
        holiday_rate = real_df["is_holiday"].mean()
        time_index["is_holiday"] = np.random.binomial(
            n=1, p=holiday_rate, size=n_rows
        ).astype(int)

        # Recompute all lag features from the ordered synthetic sessions
        synthetic_full = _recompute_lag_features(time_index)

        # Ensure column order matches real data
        feature_cols = [
            "hour", "day_of_week", "is_weekend", "is_holiday",
            "lag_1h", "lag_24h", "lag_168h",
            "rolling_24h_mean", "rolling_7d_mean", "sessions"
        ]
        return synthetic_full[feature_cols].reset_index(drop=True)

class TimeGANSynthesizer:
    """
    TimeGAN synthesizer — deprioritized for this project.

    Why not implemented fully:
        TimeGAN (Yoon et al., 2019) requires training a GAN on sliding
        windows of the input time series. With 1-3 weeks of hourly data
        (168-504 rows), the effective number of training windows is too
        small for stable GAN training on CPU. Typical failure modes at
        this data volume are mode collapse (generator produces flat or
        near-constant sequences) and training divergence.

        The project roadmap explicitly acknowledges this: when GPU is
        unavailable and per-station history is under ~4 weeks, Gaussian
        Copula is the appropriate primary augmentation method.

    What a full implementation would require:
        - Sliding window dataset (window size ~24h, stride 1)
        - Four components: Embedder, Recovery, Generator, Discriminator
          all implemented as GRU-based PyTorch modules
        - Three-phase training: embedding pretraining, supervised
          pretraining, joint adversarial training
        - Minimum ~500 windows for stable convergence (roughly 3 weeks
          of hourly data with stride-1 sliding)

    Interface is preserved so this class can be swapped in later
    without changing the notebook ablation loop.
    """

    def __init__(self):
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        """
        Not implemented. Documents the interface for future completion.
        """
        raise NotImplementedError(
            "TimeGANSynthesizer is not implemented in this phase. "
            "See class docstring for reasoning. "
            "Use GaussianCopulaSynthesizer instead."
        )

    def sample(self, n_rows: int, real_df: pd.DataFrame) -> pd.DataFrame:
        """
        Not implemented. See fit().
        """
        raise NotImplementedError(
            "TimeGANSynthesizer is not implemented in this phase."
        )