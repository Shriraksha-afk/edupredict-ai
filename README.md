# EduPredict AI — Student Performance Prediction

An end-to-end machine-learning portfolio project that estimates a student's final score from academic and study-habit inputs.

## Features
- Synthetic-data generation for reproducible experimentation
- Data preprocessing with median imputation
- Random Forest regression pipeline
- Holdout evaluation using MAE, RMSE and R²
- Interactive Streamlit prediction interface

## Important limitation
The included dataset is synthetic and the target is generated from a chosen formula plus noise. Results demonstrate implementation, not real-world predictive validity. Do not use this demo to grade, label, or make decisions about real students.

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

## Project structure
```text
edupredict-ai/
├── app.py
├── train.py
├── generate_data.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── student_performance.csv  # generated locally
└── models/
    └── student_model.joblib     # generated locally
```

## Next improvements
1. Compare Linear Regression, Random Forest and Gradient Boosting using the same split.
2. Add cross-validation and a residual plot.
3. Add feature importance with a clear note that importance is not causation.
4. Replace synthetic data with a suitable, ethically sourced dataset and document its license and limitations.
5. Add tests and a GitHub Actions workflow.

## GitHub
Create a new public repository named `edupredict-ai`, then:

```bash
git init
git add .
git commit -m "Build EduPredict AI student performance predictor"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/edupredict-ai.git
git push -u origin main
```
