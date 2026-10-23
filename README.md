# News Article Classification with Naive Bayes

A Streamlit application that trains a Multinomial Naive Bayes text classifier on the BBC news article CSV dataset and predicts the category of text entered by a user.

## Files
- `app.py` — Streamlit application and model pipeline
- `requirements.txt` — Python dependencies

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies:
   `pip install -r requirements.txt`
4. Start the app:
   `streamlit run app.py`
5. Open the local URL shown in the terminal (usually http://localhost:8501).

The app downloads the CSV from the dataset URL when it starts, so an internet connection is required.

## Deploy for free
1. Upload `app.py` and `requirements.txt` to a GitHub repository.
2. Sign in to Streamlit Community Cloud.
3. Choose **Create app**, select your repository, and set the main file path to `app.py`.
4. Deploy, then copy the public app URL for your LinkedIn post.

Dataset:
https://raw.githubusercontent.com/ronaldleonardo/Data_Transformer_Datasets/refs/heads/main/bbc_text_cls.csv
