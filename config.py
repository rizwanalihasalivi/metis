"""Central configuration for Problem F submission.

All paths are relative to the project root. Never hardcode absolute paths.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ------------------------------------------------------------------
# Top-level directories
# ------------------------------------------------------------------
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"

# ------------------------------------------------------------------
# Raw data subdirectories
# ------------------------------------------------------------------
A_DATA = DATA_RAW / "A_data_value"
B_DATA = DATA_RAW / "B_scaling_laws"
C_DATA = DATA_RAW / "C_efficiency_evolution"

# ------------------------------------------------------------------
# Q1 raw inputs
# ------------------------------------------------------------------
SLIMPAJAMA_SAMPLE = A_DATA / "slimpajama_quality_signal_sample.jsonl.xz"
SLIMPAJAMA_EXTENDED = A_DATA / "slimpajama_quality_extended"
ARXIV_RAW = SLIMPAJAMA_EXTENDED / "arxiv_part-6777d8857c6e-000486.jsonl.xz"
GITHUB_RAW = SLIMPAJAMA_EXTENDED / "github_part-6777d8857c6e-000275.jsonl.xz"

# ------------------------------------------------------------------
# Q1 processed outputs
# ------------------------------------------------------------------
Q1_DIR = DATA_PROCESSED / "q1"
A1_FEATURES = Q1_DIR / "A1_quality_features.parquet"
A1_SCALARIZED = Q1_DIR / "A1_quality_scalarized.parquet"
A1_WITH_QI = Q1_DIR / "A1_with_Qi.parquet"
ARXIV_WITH_QI = Q1_DIR / "arxiv_full_with_Qi.parquet"
GITHUB_WITH_QI = Q1_DIR / "github_full_with_Qi.parquet"

# ------------------------------------------------------------------
# Q1 model artifacts
# ------------------------------------------------------------------
Q1_MODELS = MODELS / "q1"
Q1_WEIGHTS = Q1_MODELS / "quality_weights.json"
Q1_ECDF = Q1_MODELS / "quality_ecdf.npz"
Q1_NORM = Q1_MODELS / "quality_norm.json"

# ------------------------------------------------------------------
# Q1 output directories
# ------------------------------------------------------------------
Q1_TABLES = OUTPUTS / "q1" / "tables"
Q1_FIGURES = OUTPUTS / "q1" / "figures"
Q1_CONFLICT = Q1_TABLES / "conflict"

# ------------------------------------------------------------------
# Global constants
# ------------------------------------------------------------------
SEED = 20260923
Q0 = 0.5            # Neutral quality baseline
ETA_CONTEXT = 2e-4  # Context efficiency
BUDGETS = [1e19, 1e22, 1e24]
CONTEXT_LENGTHS = [2048, 4096, 8192, 32768, 131072]

# Conflict thresholds (percentiles on A1)
CONFLICT_PERCENTILES = [0.85, 0.90, 0.95]
CONFLICT_PRIMARY = 0.90

# ==================================================================
# Q2 — B series file mapping
# ==================================================================
B1_SCALING = B_DATA / "pythia_training_log_existing.csv"
B2_CEREBRAS = B_DATA / "cerebras_training_log.csv"
B3_TRAJ_DIR = B_DATA / "training_trajectories"
B4_BASELINE = B_DATA / "scaling_baseline.csv"
B5_PUBLISHED = B_DATA / "published_scaling_data.csv"
B6_QUALITY = B_DATA / "supplementary_NQ_experiment.csv"
B7_QUALITY_EXT = B_DATA / "supplementary_NQ_experiment_expanded.csv"
B8_QUALITY_REV = B_DATA / "supplementary_NQ_experiment_large.csv"
B9_LARGE_META = B_DATA / "supplementary_large_models.csv"
B10_LARGE_LOSS = B_DATA / "supplementary_large_baseline.csv"

# ------------------------------------------------------------------
# Q2 mixture tables (A4-A15)
# ------------------------------------------------------------------
REGMIX_DIR = A_DATA / "regmix_tables"
Q2_TRAIN_MIX = REGMIX_DIR / "train_mixture_1m.csv"
Q2_TRAIN_LOSS = REGMIX_DIR / "train_pile_loss_1m.csv"
Q2_TEST_MIX = REGMIX_DIR / "test_mixture_1m.csv"
Q2_TEST_LOSS = REGMIX_DIR / "test_pile_loss_1m.csv"
Q2_60M_MIX = REGMIX_DIR / "test_mixture_60m.csv"
Q2_60M_LOSS = REGMIX_DIR / "test_pile_loss_60m.csv"
Q2_1B_MIX = REGMIX_DIR / "test_mixture_1B.csv"
Q2_1B_LOSS = REGMIX_DIR / "test_pile_loss_1B.csv"
Q2_10B_MIX = REGMIX_DIR / "est_mixture_10b.csv"
Q2_10B_LOSS = REGMIX_DIR / "est_pile_loss_10b.csv"
Q2_70B_MIX = REGMIX_DIR / "est_mixture_70b.csv"
Q2_70B_LOSS = REGMIX_DIR / "est_pile_loss_70b.csv"

# ------------------------------------------------------------------
# Q2 outputs
# ------------------------------------------------------------------
Q2_DIR = DATA_PROCESSED / "q2"
Q2_MODELS = MODELS / "q2"
Q2_TABLES = OUTPUTS / "q2" / "tables"
Q2_FIGURES = OUTPUTS / "q2" / "figures"
Q2_B1_FIT = Q2_MODELS / "b1_fit.json"
Q2_B1_BOOTSTRAP = Q2_MODELS / "b1_bootstrap.npz"
Q2_QUALITY_FITS = Q2_MODELS / "quality_fits.json"
Q2_ATTENUATION = Q2_MODELS / "attenuation.json"

# ==================================================================
# Q3 paths
# ==================================================================
C7_ARCH = C_DATA / "model_architecture_metadata.csv"

Q3_DIR = DATA_PROCESSED / "q3"
Q3_MODELS = MODELS / "q3"
Q3_TABLES = OUTPUTS / "q3" / "tables"
Q3_FIGURES = OUTPUTS / "q3" / "figures"
Q3_POINT_SOLUTIONS = Q3_MODELS / "point_solutions.json"
Q3_BOOTSTRAP = Q3_MODELS / "bootstrap_draws.csv"

# Q3 physical constants
Q3_S_DEFAULT = 4096              # primary context length
Q3_Q0 = 0.5                      # baseline quality
Q3_ETA_S = 2e-4                  # attention cost coefficient

# ==================================================================
# Q4 paths
# ==================================================================
C2_LEADERBOARD = C_DATA / "leaderboard_cleaned.csv"
C3_LEADERBOARD_ENHANCED = C_DATA / "leaderboard_enhanced.csv"
C4_EPOCH_MODELS = C_DATA / "epoch_all_ai_models.csv"
C6_BRIDGE = C_DATA / "loss_benchmark_bridge.csv"
C6_BRIDGE_EXPANDED = C_DATA / "loss_benchmark_bridge_expanded.csv"

# C8 processed
Q4_DIR = DATA_PROCESSED / "q4"
Q4_C8_TASK_METRICS = Q4_DIR / "C8_task_metrics_long.csv"
Q4_C8_PRIMARY = Q4_DIR / "C8_task_primary_scores.csv"
Q4_C8_FAMILY_LONG = Q4_DIR / "C8_family_aggregation_long.csv"
Q4_C8_MODEL_AGG = Q4_DIR / "C8_model_family_aggregate.csv"

# Q4 outputs
Q4_MODELS = MODELS / "q4"
Q4_TABLES = OUTPUTS / "q4" / "tables"
Q4_FIGURES = OUTPUTS / "q4" / "figures"
Q4_DECOMPOSITION = Q4_MODELS / "decomposition.json"
Q4_FORECAST = Q4_MODELS / "forecast.json"
Q4_BRIDGE = Q4_MODELS / "bridge_model.json"

# ==================================================================
# Utility: make all standard output directories
# ==================================================================
def ensure_dirs():
    """Create all standard output directories if missing."""
    for p in [
        DATA_PROCESSED / "q1",
        DATA_PROCESSED / "q2",
        DATA_PROCESSED / "q3",
        DATA_PROCESSED / "q4",
        Q1_MODELS,
        MODELS / "q2",
        Q3_MODELS,
        Q4_MODELS,
        Q1_TABLES,
        Q1_FIGURES,
        Q1_CONFLICT,
        OUTPUTS / "q2" / "tables",
        OUTPUTS / "q2" / "figures",
        OUTPUTS / "q3" / "tables",
        OUTPUTS / "q3" / "figures",
        OUTPUTS / "q4" / "tables",
        OUTPUTS / "q4" / "figures",
    ]:
        p.mkdir(parents=True, exist_ok=True)