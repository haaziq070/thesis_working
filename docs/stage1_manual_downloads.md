# Stage 1 — datasets that need a manual step

DARPA2000 downloads automatically via `scripts/stage1_download_darpa2000.sh`
(no registration). The two datasets below cannot be scripted from this
environment because their current hosts require you personally to submit a
form or click through a Microsoft login — there is no stable public URL to
`curl`. Do these steps yourself; they take a few minutes.

## CICIDS2017

The original CIC mirror now redirects all direct file requests back to the
info page, and the successor host (York University's BCCC lab, which now
maintains the CIC datasets) gates the download behind a request form.

1. Go to: https://www.yorku.ca/research/bccc/ucs-technical/cybersecurity-datasets-cds/dataset-request/
2. Fill in the request form (name, institutional/academic email, purpose —
   "MS thesis, alert correlation research" is sufficient). Use your own
   email for this, not mine.
3. You'll receive a download link (usually by email, sometimes immediate).
   It will point to a `.zip` or a set of per-day CSVs (the "MachineLearningCSV"
   / `GeneratedLabelledFlows` set is what we want — flow-level CSVs with
   an `Label` column).
4. Once you have the link, download it into `data/raw/cicids2017/`:
   ```bash
   cd ~/thesis
   curl -L -o data/raw/cicids2017/cicids2017.zip "<the link they gave you>"
   unzip data/raw/cicids2017/cicids2017.zip -d data/raw/cicids2017/
   ```
5. Tell me once it's there — I'll write the Stage 2 parser against whatever
   the actual folder/column layout turns out to be (CIC has changed the CSV
   schema across re-releases, so I'd rather check the real file than guess).

Note: if the form is slow to respond or you'd rather not wait, the original
CICFlowMeter-generated CSVs are also mirrored on Kaggle under "CIC-IDS2017" —
that's a viable fallback for the identifier-training data (not for anything
claimed as headline correlation ground truth, since it's a third-party
re-upload, not the canonical source). Let me know if you want that route
instead.

## UNSW-NB15 (optional, per our design discussion)

Hosted on a UNSW OneDrive/SharePoint folder, which requires a browser session
(no anonymous scripted download):

1. Go to: https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. Follow either of the two "download" links on that page (they open a
   SharePoint folder view).
3. In the browser, select the CSV files (the four `UNSW-NB15_*.csv` parts
   plus `NUSW-NB15_features.csv` and `NUSW-NB15_GT.csv`) and download.
4. Move them into `data/raw/unsw-nb15/` on this machine, e.g. via `scp` from
   your laptop:
   ```bash
   scp UNSW-NB15_*.csv NUSW-NB15_*.csv user@this-host:~/thesis/data/raw/unsw-nb15/
   ```

Since UNSW-NB15 is only an optional secondary generalization check (not
required by the core pipeline), we can proceed with Stage 1 and come back to
this later — don't let it block progress.
