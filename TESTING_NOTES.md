# Testing Notes

Use the `calp` conda environment for test runs.

## Quick checks

```bash
conda run -n calp python -m pytest --version
```

```bash
conda run -n calp python -m pytest -q
```

## Targeted maxent_large tests

```bash
conda run -n calp python -m pytest -q tests/maxent_large
```

```bash
conda run -n calp python -m pytest -q tests/maxent_large/test_distribution_builder_smoke.py
```

```bash
conda run -n calp python -m pytest -q tests/maxent_large/test_multi_condition_probability.py
```

```bash
conda run -n calp python -m pytest -q tests/maxent_large/test_conditional_expectation.py
```

## If collection/import fails

Run from repo root:

```bash
cd /home/amdson/calibrated_response
```

If needed, install project deps in `calp`:

```bash
conda run -n calp python -m pip install -e .
```
