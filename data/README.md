# Data source

The raw transaction file is intentionally excluded from Git because it is about 44 MB.

The analysis first looks for a local CSV in `data/raw/`. If none is available, download the course-approved fallback:

```bash
curl -L \
  https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/07_Visualization/Online_Retail/Online_Retail.csv \
  -o data/raw/Online_Retail.csv
```

The pipeline treats the local file as the source of truth and records its SHA256 checksum, dimensions, schema, missingness, and date range in `outputs/tables/dataset_audit.csv` and `outputs/tables/dataset_metadata.csv`.

Verified local file:

- Filename: `Online_Retail.csv`
- Shape: 541,909 rows × 8 columns
- Date range: 2010-12-01 08:26 to 2011-12-09 12:50
- SHA256: `5c1b5517919301b1da060b3dc486614f487da43515a9b2a52709e2b04d5da575`

Dataset provenance: UCI Machine Learning Repository, *Online Retail* transactional dataset. Currency is GBP (£).
