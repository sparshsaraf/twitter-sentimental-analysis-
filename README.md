# Twitter Sentiment Analysis using Python

This project performs sentiment analysis on Twitter data using machine learning. It utilizes logistic regression to classify tweets as either positive or negative. The dataset is obtained from Kaggle and processed using NLP techniques.

---

## Features

- **Automated Data Import**: Uses the Kaggle API to fetch large datasets efficiently.
- **Data Preprocessing**: Cleans and processes tweet text using stemming and stopword removal.
- **Feature Extraction**: Converts text into numerical representations using TF-IDF vectorization.
- **Model Training & Evaluation**: Trains a logistic regression model and evaluates its accuracy.
- **Real-time Sentiment Prediction**: Allows users to input text and predict sentiment.

---

## Prerequisites

Ensure you have Python installed. Install the required libraries using the following command:

```bash
pip install kaggle numpy pandas nltk scikit-learn
```

You will also need to set up Kaggle API authentication:

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## Project Structure

```bash
.
├── main.py
├── README.md
├── requirements.txt
└── trained_model.sav
```

- `main.py` contains the code for data preprocessing, model training, and sentiment prediction.
- `README.md` provides project documentation.
- `requirements.txt` lists the required dependencies.
- `trained_model.sav` is the saved logistic regression model.

---

## Usage

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/twitter-sentiment-analysis.git
```

2. **Navigate to the project folder:**
```bash
cd twitter-sentiment-analysis
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Fetch dataset from Kaggle:**
```bash
kaggle datasets download -d kazanova/sentiment140
```

5. **Extract dataset:**
```bash
unzip sentiment140.zip
```

6. **Run the script:**
```bash
python main.py
```

7. **Predict Sentiment:**
```bash
Enter a sentence to analyze sentiment: "I love Python!"
Output: "The sentiment is positive."
```

---

## Model Training

- Uses **Logistic Regression** with `max_iter=1000` for better convergence.
- Evaluates model accuracy on training data.
- Saves the trained model as `trained_model.sav` for later use.

---

## Troubleshooting

- **Kaggle API Issues**: Ensure `kaggle.json` is correctly placed and permissions are set.
- **NLTK Stopwords Not Found**: Run `nltk.download('stopwords')` before executing the script.
- **Module Not Found**: Run `pip install -r requirements.txt` again.

---


## Acknowledgments

- [Kaggle Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
- [NLTK](https://www.nltk.org/) for text preprocessing.
- [Scikit-learn](https://scikit-learn.org/) for machine learning.
- [Pandas](https://pandas.pydata.org/) for data manipulation.

