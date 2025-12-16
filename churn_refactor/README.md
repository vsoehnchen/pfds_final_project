# Churn Project Refactor (Notebook -> Modules)

## Install (editable)
From the project root:

```bash
pip install -e .
```

## How to use from a notebook
```python
from churnproj.features.build import build_dual_window_features
from churnproj.modeling.split import SnapshotTimeSplit
```

## Run example script
```bash
python scripts/run_pipeline.py --data path/to/logs.parquet
```

## Notes
- Keep exploratory work in `notebooks/`.
- Put reusable logic in `src/churnproj/`.