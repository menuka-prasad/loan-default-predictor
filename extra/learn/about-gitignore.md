### The reasoning for not commiting data folder into git:

1. Size. Git is built to diff and version text, efficiently. A CSV with 150K rows is a blob Git can't meaningfully diff — every change just stores a whole new copy. Repos bloat fast and become painful to clone.
2. Reproducibility over storage. Best practice isn't "don't track data," it's "don't store raw data as opaque blobs in your main version history." You instead version how to get the data (a script, a DVC pointer, a documented source) so anyone — including future you — can reproduce it without your repo turning into a data warehouse.
3. Secrets/PII risk. Real data, even semi-real Kaggle data, sometimes shouldn't be public. Habit of gitignoring data/ by default protects you later when the data does matter.

**What you should actually do this time, properly:**
data/
├── raw/          # gitignored — original Kaggle download, untouched
├── processed/    # gitignored — cleaned/feature-engineered versions
└── README.md     # NOT gitignored — says: "download from <kaggle link>, run scripts/prepare_data.py"\

So the data/ folder itself is committed (it has a README), but its contents (the actual CSVs) are gitignored. That's the distinction most beginners miss — they either commit everything or gitignore the whole folder including the explanation of what should go there.