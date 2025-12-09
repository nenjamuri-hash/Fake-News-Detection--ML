

##  Fake News Detection (Machine Learning + Google Verification)

This project detects whether a news headline is **Real** or **Fake** using:

* A **Transformer-based ML model (RoBERTa)** trained on the **LIAR dataset**
* **Google News RSS verification** by checking real-world news overlap
* A **Streamlit app** for simple web-based use
* Supports **single prediction**, **CSV prediction**, and **Excel prediction** uploads

This tool helps quickly check misleading news content and misinformation online.

---

##  Features

| Feature             | Description                             |
| ------------------- | --------------------------------------- |
| ML-Based Prediction | Classifies headline as REAL or FAKE     |
| Google Verification | Checks if similar news exists online    |
| Combined Score      | Weighted model + Google result          |
| CSV Upload          | Batch prediction for multiple headlines |
| Excel Upload        | Same as CSV, works with `.xlsx`         |

---

##  Folder Structure (as in this repository)

```
Fake-News-Detection-ML/
│
├── config.py           
├── data_loader.py      
├── evaluate.py         
├── import_matplot.py   
├── inference.py        
├── predict.py          
├── preprocess.py       
├── streamlit_app.py     ← RUN THIS FOR UI
├── train_deberta.py    
├── train_roberta.py    
├── train_utils.py      
├── utils.py            
├── utils_metrics.py    
├── verify_with_google.py
├── batch_predict.py     
├── TEST.py              
└── README.md
```

> The **trained model folder** should be placed inside `/artifacts/model/`
> Name: **model** (required for inference)

| File/Folder             | Purpose                                          |
| ----------------------- | ------------------------------------------------ |
| `streamlit_app.py`      | Runs the online Fake News detection UI           |
| `verify_with_google.py` | Checks if similar headlines exist on Google News |
| `train_roberta.py`      | Trains the RoBERTa model                         |
| `evaluate.py`           | Measures accuracy of the model                   |
| `batch_predict.py`      | Predicts FAKE/REAL for bulk CSV or Excel file    |
| `preprocess.py`         | Cleans text before training and predictions      |
| `utils_metrics.py`      | Calculates performance metrics                   |
| `artifacts/model`       | Stores the trained ML model                      |
| `artifacts/tokenizer`   | Stores tokenizer for encoding text               |

---

##  Dataset Used

The project uses **LIAR Dataset (~12,836 labeled political statements)**
This dataset contains headlines marked as: *true, half-true, false, pants-on-fire, etc.*
We simplified them into **REAL/FAKE categories** for binary classification.

---

##  How to Run the Streamlit App

### 1. Install requirements

```
pip install -r requirements.txt
```

### 2️. Run Streamlit UI

```
streamlit run streamlit_app.py
```

### 3. Make Predictions

You can:

✔ Type a headline
✔ Upload **CSV** (column name: `headline`)
✔ Upload **Excel (.xlsx)** (column name: `headline`)

The result file can be downloaded after prediction.

---

## Dependencies

* Python 3.9+
* PyTorch
* Transformers
* Pandas
* Feedparser
* Streamlit

---

##  Example Input Format (CSV or Excel)

| headline                         |
| -------------------------------- |
| Canada increases student visas   |
| Aliens will attack next week     |
| India and US sign new trade deal |

---

## Output File Includes:

| headline | model_label | google_score | final_verdict | reason |

---

## Why This Project
Fake news spreads quickly and misleads people.
This project provides a **practical and simple way** to verify authenticity.


