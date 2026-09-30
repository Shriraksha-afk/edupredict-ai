
# EduPredict AI — Student Performance Prediction

An end-to-end machine-learning portfolio project that estimates a student's final score from academic and study-habit inputs.

## Features

- Synthetic-data generation for reproducible experimentation
- Data preprocessing with median imputation
- Random Forest regression pipeline
- Holdout evaluation using MAE, RMSE and R²
- Interactive Streamlit prediction interface

## Important Limitation

The included dataset is synthetic, and the target is generated from a chosen formula plus noise. Results demonstrate implementation, not real-world predictive validity.

Do not use this demo to grade, label, or make decisions about real students.

## Model Evaluation

The Random Forest regression model is evaluated using a holdout test split.

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| R² | Coefficient of Determination |

Evaluation results should be added here after running `train.py`.

## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py
python train.py
streamlit run app.py
```

Open the local URL shown by Streamlit.

## Project Structure

```text
edupredict-ai/
├── app.py
├── auth.py
├── train.py
├── generate_data.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── student_performance.csv
└── models/
    └── student_model.joblib
```

The dataset and trained model are generated locally and may be excluded from GitHub.

## Next Improvements

1. Compare Linear Regression, Random Forest and Gradient Boosting using the same split.
2. Add cross-validation and a residual plot.
3. Add feature importance with a clear note that importance is not causation.
4. Replace synthetic data with a suitable, ethically sourced dataset and document its license and limitations.
5. Add tests and a GitHub Actions workflow.

## GitHub

Repository: [EduPredict AI](https://github.com/Shriraksha-afk/edupredict-ai)

To upload updates to the existing repository:

```bash
git add app.py train.py README.md
git commit -m "Update EduPredict AI project"
git push origin main
```