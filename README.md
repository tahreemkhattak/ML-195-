# Full-Stack Machine Learning App

This project meets the assignment requirements by training two scikit-learn models, saving them with `joblib`, serving predictions through FastAPI POST endpoints, and providing a one-page HTML frontend.

## Models
1. **Regression:** `LinearRegression` trained on scikit-learn's Diabetes dataset. It predicts a numeric disease-progression measure from 10 standardized features. Evaluation metric: R².
2. **Classification:** `LogisticRegression` trained on scikit-learn's Iris dataset. It predicts Setosa, Versicolor, or Virginica from four flower measurements. Evaluation metric: accuracy.

The training script prints the actual R² and accuracy calculated from a held-out test set. Report those printed values in your submission rather than guessing them.

## Project structure

```text
full_stack_ml_app/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── models/                 # created after training
└── static/
    └── index.html
```

## Run on Windows

1. Install Python 3.10 or newer.
2. Open Command Prompt or the VS Code terminal in this project folder.
3. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

4. Activate it:

   ```bash
   .venv\Scripts\activate
   ```

5. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

6. Train and save the models:

   ```bash
   python train_models.py
   ```

   Copy the printed **Regression R²** and **Classification accuracy** values into your report. This also creates the `models` folder with `.joblib` files.

7. Start the web server:

   ```bash
   uvicorn app:app --reload
   ```

8. Open `http://127.0.0.1:8000` in your browser.

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## Endpoints

### `POST /predict/regression`

Example JSON body (10 features):

```json
{
  "age": 0.04,
  "sex": 0.05,
  "bmi": 0.06,
  "bp": 0.02,
  "s1": -0.04,
  "s2": -0.03,
  "s3": -0.01,
  "s4": 0.02,
  "s5": 0.03,
  "s6": 0.01
}
```

Returns JSON such as:

```json
{"prediction": 150.0}
```

The numeric value above is only an illustrative example; the real prediction depends on the trained model and input.

### `POST /predict/classification`

Example JSON body:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Returns a JSON object with `class_id` and the predicted species, for example:

```json
{"class_id": 0, "prediction": "setosa"}
```

## How it works

- `train_models.py` loads the datasets, splits each into training and testing data, trains the models, evaluates them, and saves the models using `joblib.dump`.
- `app.py` loads the saved models and defines the two POST endpoints. Each endpoint validates incoming JSON and returns the prediction as JSON.
- `static/index.html` contains a form for each model. JavaScript sends form values to the API with `fetch()` and displays the returned prediction.

## GitHub submission

Create a new repository on GitHub, then run these commands in the project folder (replace the URL with your own repository URL):

```bash
git init
git add app.py train_models.py requirements.txt README.md static/
git commit -m "Build full-stack machine learning app"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

After training, you can also add the `models/` folder to the repository if your instructor expects the saved model files to be submitted. Do not upload your virtual environment (`.venv`).
