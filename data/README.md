# Data source

The raw transaction file is intentionally excluded from Git because it is about 44 MB.

The analysis first looks for a local CSV in `data/raw/`. If none is available, download the course-approved fallback:

```bash
curl -L \
  https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/07_Visualization/Online_Retail/Online_Retail.csv \
  -o data/raw/Online_Retail.csv
```

The pipeline treats the local file as the source of truth and records its SHA256 checksum, dimensions, schema, missingness, and date range in `outputs/tables/dataset_audit.csv` and `outputs/tables/dataset_metadata.csv`.

Dataset provenance: UCI Machine Learning Repository, *Online Retail* transactional dataset. Currency is GBP (£).

