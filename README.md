## Digger-Guider: Daily-Frequency Stock Trend Prediction

This repository now focuses solely on daily-bar features (one feature vector per trading day). All high-frequency resampling and dual-branch high/low-frequency modeling have been removed. The pipeline loads daily factors, trains the daily models, and runs backtesting with the updated scripts.

### 0. Dependencies

Install Python packages from `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 1. Prepare Daily Data (via qlib)

All training relies on the daily dataset (e.g., `day_csi300`). Intraday resampling is no longer needed.

```bash
cd load_data
python load_dataset.py
```

* Adjust `market` inside `load_dataset.py` to switch datasets (e.g., `csi300`, `NASDAQ`).
* The default daily feature window is 20 trading days with 6 base factors (OPEN, CLOSE, HIGH, LOW, VOLUME, VWAP), producing `6 * 20 + 1` columns including the label.

> **Note:** `high_freq_resample.py` has been disabled in the codebase; running it is no longer required for training.

### 2. Framework Overview

All models live in `./framework/models/cnn_rnn_v2.py` and now consume only daily sequences:

* **Day_Model_1**: RNN over the daily feature window to produce latent representations and preliminary predictions.
* **Day_Model_2**: RNN that refines predictions by combining Day_Model_1 hidden states with raw daily inputs.

Mutual-distillation components and mixed-frequency branches have been removed; training is a single daily-only pipeline.

### 3. Train

```bash
cd framework
python main_cnn_rnn_v2.py with config/main_model.json model_name=cnn_rnn_v2
```

Key runtime knobs (editable via CLI overrides or `config/main_model.json`):

* **Data window**: `daily_loader_v3.pre_n_day` (default 20)
* **Input shape**: `cnn_rnn_v2.input_shape` (default `[6, 20]` → 6 factors over 20 days)
* **Training splits**: date ranges in `daily_loader_v3`
* **Optimization**: learning rates in `cnn_rnn_v2.optim_args` / `optim_args_2`

Model checkpoints and predictions are written to `./framework/out/` (path configurable via `output_path`).

### 4. Market Trading Simulation

Prerequisites:

* qlib server with the corresponding market data
* Prediction results generated from the training step

Run the simulator:

```bash
cd framework
python trade_sim.py
```

### 5. Experiment Records

Each experiment creates a record under `./framework/my_runs/` containing:

* `config.json`: parameter settings and data paths
* `cout.txt`: detailed model output and results
* `pred_{model_name}_{seed}.pkl`: serialized predictions (`score`) and labels
* `run.json`: script hashes; source snapshots stored under `./framework/my_runs/source/`

Use these artifacts to reproduce or audit runs after the daily-only refactor.
