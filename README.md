# Smart Irrigation Prediction — ML + CI/CD

Predicts irrigation need (Low / Medium / High) from soil, crop, and weather
data, served as an API, with an automated CI/CD pipeline on GitHub Actions.

## What's in this project

| File | Purpose |
|---|---|
| `data/irrigation_prediction.csv` | Your dataset (10,000 rows) |
| `train_model.py` | Trains **8 models**, compares them, saves the best one as `model.joblib` |
| `app.py` | FastAPI service that loads the best model and serves `/predict` |
| `test_app.py` | Automated tests (run by CI on every push) |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Packages the trained model + API into a container |
| `.github/workflows/ci-cd.yml` | The CI/CD pipeline itself |
| `.gitignore` / `.dockerignore` | Keep generated files out of Git and out of the Docker image |
| `.flake8` | Settings for the code-style check that runs in CI |

## Prerequisites (install these first)

1. **Python 3.10+** — check with `python3 --version`
2. **Git** — check with `git --version`
3. **A GitHub account** and an empty repo created on github.com
4. **Docker Desktop** (only needed if you want to test the container locally —
   not required for the CI/CD pipeline to work, GitHub does the building)

## Step-by-step: get it running locally

```bash
# 1. Go into the project folder
cd irrigation-project

# 2. Create a virtual environment (keeps dependencies isolated)
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train and compare all 8 models
python train_model.py
# Trains Logistic Regression, Decision Tree, KNN, SVM, Random Forest,
# XGBoost, LightGBM and CatBoost; prints a ranked comparison; saves the
# best one (by macro-F1) as model.joblib

# 5. Run the API
uvicorn app:app --reload

# 6. Open http://127.0.0.1:8000/docs in your browser
#    Click /predict -> "Try it out" -> it's pre-filled with an example -> Execute
```

If step 6 gives you a prediction back, your model + API are working end to end.

## Model comparison results

| Model | Accuracy | Macro-F1 | Recall on "High" class |
|---|---|---|---|
| **CatBoost** (chosen) | 99.95% | 99.73% | 1.00 |
| XGBoost | 99.75% | 99.11% | 0.97 |
| Decision Tree | 99.65% | 98.78% | 0.94 |
| LightGBM | 99.50% | 98.16% | 0.91 |
| Random Forest | 97.25% | 82.37% | 0.34 |
| SVM (RBF) | 86.55% | 76.80% | 0.58 |
| Logistic Regression | 78.30% | 68.15% | 0.82 |
| K-Nearest Neighbors | 74.85% | 54.01% | 0.09 |

**The interesting finding**: tree/boosting-based models (CatBoost, XGBoost, Decision Tree,
LightGBM) all score 98%+ macro-F1 — including near-perfect recall on the rare "High"
class that plain Random Forest struggled with (34% recall). Linear and distance-based
models (Logistic Regression, KNN, SVM) score much lower. This suggests the relationship
between the input features and irrigation need is **non-linear and threshold-driven**
rather than linear — exactly what tree-based splits capture well and straight-line
decision boundaries don't. `train_model.py` picks the best model automatically by
macro-F1, so `model.joblib` is always CatBoost unless you change the data.

Full per-model, per-class numbers are in `model_comparison.json` after training.

## Version control (Git) — already set up

This folder is **already a Git repository** with a clean commit history
(run `git log --oneline` to see it). A proper `.gitignore` is included so
generated files (`model.joblib`, `venv/`, etc.) never get committed.

**One-time setup — connect it to GitHub:**

1. Create a new **empty** repository on github.com (no README, no .gitignore).
2. Tell Git who you are (this is what shows up as the commit author):
   ```bash
   git config user.name "Your Name"
   git config user.email "you@example.com"
   ```
3. Connect and push (do **not** run `git init` again):
   ```bash
   git remote add origin <your-empty-github-repo-url>
   git push -u origin main
   ```

Check the **Actions** tab on your GitHub repo — the pipeline starts by itself.

### How the team should work (so nobody overwrites anyone's work)

```bash
git checkout -b feature/add-weather-api     # 1. make your own branch
# ... edit files ...
git add .
git commit -m "Add weather API input"       # 2. save your work with a clear message
git push -u origin feature/add-weather-api  # 3. push your branch
```

Then open a **Pull Request** on GitHub. The CI pipeline runs automatically on
the pull request — if it shows a green tick, your teammate reviews and merges
it into `main`. If it shows a red cross, fix it before merging. This is what
"version control + CI" means in practice: nobody can break `main` without the
tests catching it.

## What the CI/CD pipeline does

| Stage | Runs on | What happens |
|---|---|---|
| **CI** — `test` job | every push and every pull request | Installs dependencies → lints the code (`flake8`) → trains and compares all 8 models → runs the tests → saves the model + results as downloadable artifacts |
| **CD** — `docker-build-and-push` job | only pushes to `main`, and only if CI passed | Builds the Docker image → **starts the container and calls `/health` and `/predict`** to prove it really works → only then pushes it to GitHub's container registry (`ghcr.io`), tagged `latest` and with the commit id |

No secrets or extra accounts are needed — GitHub provides the login token automatically.

## What NOT to do

- **Don't commit `venv/`** or your trained `model.joblib` if it's large —
  the pipeline regenerates it. The included `.gitignore` already handles this.
- **Don't hardcode API keys or passwords** anywhere in the code. This
  project doesn't need any, which is intentional — keep it that way.
- **Don't only report accuracy.** Your "High" irrigation-need class is only
  3.4% of the data, so accuracy alone is misleading (a model that never
  predicts "High" would still score ~96%). Report macro-F1 and the
  per-class recall too — `metrics.json` has this after training.
- **Don't skip the tests.** If `pytest` fails, the Docker image never gets
  built — that's the pipeline doing its job, not a bug.

## Talking points for your report / viva

- **Problem framing**: multi-class classification (Low/Medium/High
  irrigation need) from 19 soil, crop, and weather features — consistent
  with the "multi-parameter predictive irrigation" direction your
  literature survey identifies (SIDSS, PLSR/ANFIS studies).
- **Model choice**: Random Forest — chosen for robustness on mixed
  categorical/numeric tabular data and resistance to overfitting, over
  deep learning (which your literature survey notes needs much larger,
  denser sensor datasets than you have here).
- **Class imbalance**: addressed with `class_weight="balanced"`; still
  a limitation worth discussing openly (see metrics above) — this mirrors
  the "insufficient sensor density / limited observation period"
  limitations your literature survey flags in the RNN-LSTM study.
- **Deployment**: model wrapped in a FastAPI service, containerized with
  Docker, and shipped through an automated CI/CD pipeline (GitHub Actions)
  that tests before it builds — a genuine (if minimal) MLOps pipeline,
  not just a notebook.

## Sensor feasibility (from the hardware/mapping docs)

Cross-checking the model's 19 input features against your team's actual hardware
(ESP32 + sensors) and the sensor-to-CSV mapping doc:

| Status | Features | Notes |
|---|---|---|
| **Live sensor, ready now** | Soil_Moisture, Temperature_C, Humidity | Soil moisture sensor + DHT11/DHT22 |
| **Sensor available, needs logic** | Rainfall_mm, Sunlight_Hours, Wind_Speed_kmh | Rain sensor/LDR give proxy signals (yes/no, intensity) — combine with a Weather API for the real mm/hours/km-h values the model expects |
| **On hold / future hardware** | Soil_pH, Electrical_Conductivity | pH sensor is built but not yet wired in; EC needs the (future) NPK sensor or a dedicated EC sensor |
| **No sensor — manual/config input** | Soil_Type, Crop_Type, Crop_Growth_Stage, Season, Irrigation_Type, Water_Source, Field_Area_hectare, Mulching_Used, Region, Previous_Irrigation_mm | Set once per field/crop cycle via an app or config file — not something a sensor measures |
| **Not a feature — model output** | Irrigation_Need | This is what the model predicts, not an input |
| **Not currently in the model** | Organic_Carbon | No sensor or lab-test source identified yet — you may want to drop it or gather it separately if going to real deployment |

**What this means for your demo**: you can honestly say the model is trained on a
superset of features, some of which are live-sensor-ready today and some of which are
roadmap items — the mapping to your actual ESP32 hardware is already documented, so
the gap between "the trained model" and "the real deployed device" is a known, scoped
piece of future work rather than an oversight.
