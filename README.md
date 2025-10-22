# Fake News Classifier (Bi-LSTM)

Small Streamlit app that classifies news text as Fake or Real using a trained Keras model.

## Contents
- app/main.py — Streamlit app (loads model from `app/`)
- app/requirements.txt — runtime dependencies
- (model files are NOT included in repo by default)

## Quickstart (Windows CMD)
1. Open project root:
   cd "C:\fakenewsby bit lstm"

2. Create & activate venv:
   py -3 -m venv .venv
   .\.venv\Scripts\activate

3. Install deps:
   pip install --upgrade pip
   pip install -r "app\requirements.txt"

4. Ensure model is present:
   - Preferred: host the model externally (GitHub Release, S3) and let the app download it.
   - Or copy your HDF5 model into `app\fake_news_model.h5` or place a SavedModel folder at `app\saved_model`.

5. Run the app:
   streamlit run "app\main.py"

## Recommended model hosting (keep repo small)
- Upload `fake_news_model.h5` as a GitHub Release asset or to S3/GCS.
- In `app/main.py` set `MODEL_URL` to the asset URL; the app can download it at startup if the file is missing.
- Do NOT commit large model files to git; use Git LFS only if necessary.

Example downloader pattern (already referenced in app):
```python
MODEL_URL = "https://github.com/<USER>/<REPO>/releases/download/v1.0/fake_news_model.h5"
if not os.path.exists(MODEL_PATH):
    urlretrieve(MODEL_URL, MODEL_PATH)
```

## Tokenizer
- If you trained with a Keras Tokenizer, save it and place in `app/` so inference indices match training:
  - Pickle: `pickle.dump(tokenizer, open("app/tokenizer.pickle","wb"))`
  - JSON: `open("app/tokenizer.json","w").write(tokenizer.to_json())`

## Git & Deploy
- Add a `.gitignore` that excludes `.venv/`, model files, and tokenizer files.
- Init & push:
  git init
  git add .
  git commit -m "Initial commit"
  git remote add origin https://github.com/<you>/<repo>.git
  git push -u origin main

- Deploy to Streamlit Cloud:
  1. Sign in at https://streamlit.io/cloud
  2. Create a new app → select repo, branch, and `app/main.py`
  3. Add secrets if your model URL is private

## Troubleshooting
- Blank page: check Streamlit terminal for stack traces.
- Embedding errors (index out of range): ensure tokenizer/voc_size used in inference matches training; use the saved tokenizer or clip indices before predict.
- If `Could not load model` appears, confirm `app/fake_news_model.h5` or `app/saved_model/` exists or that the downloader URL is correct.

## Author
Project by Srajal Tiwari

## License
MIT (adjust as needed)