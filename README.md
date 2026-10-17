# AI-Powered Student Performance Analysis and Prediction System

Data Analysis with AI internship project by **Dhruv (Modi Dhruv Bhaveshbhai)**.

An end-to-end data analysis and machine learning project that explores what is associated with students' final grades and builds a model that **estimates** the final grade.

## Dataset

- **Name:** Student Performance (Mathematics course, `student-mat.csv`)
- **Source:** UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/320/student+performance
- **Size:** 395 students × 33 columns (0 missing values, 0 duplicates)
- **Target:** `G3` — final grade (0–20)
- **Licence:** listed on the UCI page as CC BY 4.0 (attribution required) — re-check the page before redistributing
- **Citation:** P. Cortez and A. Silva, *Using Data Mining to Predict Secondary School Student Performance*, FUBUTEC 2008
- **Not in the dataset:** attendance percentage and exact study hours. Nothing was invented; `absences` and the `studytime` group are used instead.
- **Loading:** the code tries the official UCI zip, then a public mirror of the same file. If both fail, download `student-mat.csv` manually and place it in `data/raw/`. A copy (about 50 KB) is already included.

## What the project does

- **Cleaning:** validates every column against the data dictionary; flags (does not blindly delete) outliers
- **EDA and statistics:** 12 charts, Spearman correlations, Mann–Whitney / Kruskal–Wallis tests, effect sizes, a confounding check
- **Modelling:** Dummy baseline, Linear Regression, Random Forest, Gradient Boosting; leakage-safe scikit-learn pipelines; 5-fold CV tuning; one final test evaluation
- **Two scenarios:** (A) with earlier grades G1/G2, (B) "early warning" without any grades
- **Prediction function:** validates input, uses the saved model, returns an estimate
- **Dashboard:** 6-page Streamlit app
- **Fairness:** `sex`, `address`, `famsize`, `Pstatus`, `Mjob`, `Fjob`, `guardian`, `romantic`, `nursery`, `reason` are excluded from the models

## Key results (test set = 79 students, seed 42)

| Scenario | Selected model | MAE | RMSE | R² |
|---|---|---|---|---|
| A – with G1, G2 | Gradient Boosting | 1.06 | 1.76 | 0.85 |
| B – no grades | Random Forest | 3.27 | 3.97 | 0.23 |
| Baseline (predict the mean) | Dummy | 3.65 | 4.55 | −0.01 |

- Earlier grades make estimates fairly accurate; habits and background alone do **not**.
- All 38 students with `G3 = 0` have 0 absences — probably students who did not sit the final assessment (an inference, not a confirmed fact).
- Study time shows no clear pattern once earlier grades are held roughly constant.
- The test set is small, so small differences between models are not meaningful. Associations are not causation; predictions are estimates, not guarantees.

## Technologies

Python · Pandas · NumPy · SciPy · Matplotlib · Seaborn · Scikit-learn · Joblib · Streamlit · Jupyter · pytest · Git/GitHub

## Setup and run

```bash
git clone https://github.com/YOUR-USERNAME/Student-Performance-Analysis.git
cd Student-Performance-Analysis
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

- **Notebook (main file):** `jupyter notebook notebooks/Dhruv_StudentPerformanceAnalysis.ipynb` → Run All
- **Scripts (same logic):**
  ```bash
  python -m src.data_cleaning     # clean data -> data/processed/
  python -m src.eda               # charts -> visualizations/, tables -> reports/tables/
  python -m src.train_model       # models -> models/, metrics -> models/metrics.json
  python -m src.predict           # demo prediction
  ```
- **Dashboard:** `streamlit run dashboard/app.py`
- **Tests:** `pytest -q` (11 tests)

## Repository structure

```
Student-Performance-Analysis/
├── data/raw/ , data/processed/
├── notebooks/Dhruv_StudentPerformanceAnalysis.ipynb
├── src/            config.py, data_cleaning.py, eda.py, train_model.py, predict.py
├── models/         model.joblib, model_early_warning.joblib, metrics.json
├── visualizations/ 18 PNG charts
├── dashboard/app.py
├── reports/        Dhruv_ProjectReport.docx, tables/
├── tests/test_data_validation.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Limitations

- 395 students, two schools, one subject — results may not generalise
- No attendance percentage or exact study hours
- Likely dropouts (`G3 = 0`) are hard to predict
- Model A mostly says "the final grade will resemble the last grade"

## Ethical use

- Use estimates to start supportive conversations, never to label, group or penalise students.

## Author

Dhruv (Modi Dhruv Bhaveshbhai) — B.Tech Mechanical Engineering, Government Engineering College, Patan; B.Sc. AI & Data Science, IIT Jodhpur.
