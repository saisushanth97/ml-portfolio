# 30-Day Hands-On ML Plan (45 min/day)

**Real goal this plan serves:** one real, defensible end-to-end ML project — built by your own hands, not copied from a tutorial — that you can talk about fluently in an interview. Not "AI/ML expert." That claim in 30 days is not honest, and you've never wanted fabrication.

**Non-negotiable rule for all 30 days:** use ONE dataset the whole way through (pick something banking/finance-flavored so it doubles as an interview story — e.g. a credit/loan default or fraud dataset from Kaggle). Do not switch datasets mid-way. Do not follow a YouTube tutorial that writes the code for you — type every line yourself, even if it's slower.

---

## Week 1 — Data foundations (no modeling yet)

**Day 1 — Environment + first look**
Task: Install Python, Jupyter, pandas, scikit-learn. Create a GitHub repo `ml-portfolio`. Load your chosen dataset, run `.shape`, `.info()`, `.describe()`, `.isnull().sum()`.
Done right if: notebook runs top-to-bottom with zero errors; you can state row count, column count, and which columns have missing values, out loud, without looking.
Don't: don't touch modeling. Don't read "what is machine learning" articles — you already know the theory.

**Day 2 — Clean the data**
Task: Handle missing values and duplicates. For each column with missing data, decide and write (in a markdown cell) *why* you dropped/imputed it.
Done right if: `.isnull().sum()` is zero on all columns you're keeping; every decision has one written sentence of justification.
Don't: don't use a fancy imputation library. Simple mean/median/mode or drop is fine.

**Day 3 — Look at the data**
Task: Plot the target variable's distribution + 3 key feature distributions (matplotlib/seaborn).
Done right if: 4 plots exist, each with one written insight ("X is heavily skewed, may need log transform").
Don't: don't spend time on plot styling/colors. Ugly plots are fine.

**Day 4 — Make it model-ready**
Task: Encode categorical columns (one-hot or label encoding), scale numeric columns.
Done right if: your feature matrix `X` is 100% numeric, has zero NaNs, and shape makes sense.
Don't: don't use pipelines or ColumnTransformer yet — do it manually so you understand what's happening.

**Day 5 — First model, however bad**
Task: `train_test_split`, fit a `LogisticRegression`, call `.predict()`, print accuracy.
Done right if: accuracy beats the "always predict majority class" baseline.
Don't: don't tune anything. A bad-but-working model is the win today.

**Day 6 — The metric that actually matters (this is the gap that hurt you before)**
Task: For 10 sample predictions, manually build a confusion matrix on paper/markdown, then compute precision/recall/F1 by hand. Then run `sklearn.metrics.classification_report` and confirm it matches.
Done right if: your hand-calculated numbers match sklearn's exactly.
Don't: don't skip the manual step and jump to sklearn. The manual step is the whole point — this is exactly what broke down under pressure last time.

**Day 7 — Week 1 stress test**
Task: Restart the kernel, run the entire notebook top to bottom with zero manual fixes mid-run.
Done right if: it completes in under 2 minutes with no errors.
Don't: don't add any new code today.

---

## Week 2 — Core algorithms, understood not memorized

**Day 8 — Decision Tree**
Task: Fit `DecisionTreeClassifier`, use `plot_tree` to visualize top 3 levels, write in your own words why the top split was chosen.
Done right if: you can explain the top split using gini/entropy without looking anything up.

**Day 9 — Random Forest**
Task: Fit `RandomForestClassifier`, compare accuracy to the single tree, plot `feature_importances_`.
Done right if: you can name the top 5 features and explain in one sentence why RF beats a single tree (bagging + averaging).
Don't: don't tune hyperparameters yet.

**Day 10 — Overfitting, made visible**
Task: Loop `max_depth` from 1 to 20, plot train accuracy vs test accuracy on the same chart.
Done right if: you can point to the exact depth where the lines diverge and say "that's overfitting starting."

**Day 11 — Cross-validation**
Task: Run `cross_val_score` with 5 folds on your best model so far. Compare mean/std to your single train/test split result.
Done right if: you can explain in one sentence why the CV score is more trustworthy than one split.

**Day 12 — Hyperparameter tuning**
Task: `GridSearchCV` on Random Forest, tuning only `n_estimators` and `max_depth` (nothing else — keep it small).
Done right if: `best_params_` prints, and the tuned score beats your Day 9 baseline.
Don't: don't tune 6 hyperparameters at once. Two is the whole task.

**Day 13 — Gradient Boosting**
Task: Fit `XGBoost` or `GradientBoostingClassifier`. Compare to Random Forest.
Done right if: model trains without error and you have a metric to compare.

**Day 14 — Week 2 review**
Task: Build one markdown table comparing all 4 models (LogReg, Tree, RF, GB) on accuracy/precision/recall/F1.
Done right if: the table exists and you can say which model wins and why, in one sentence.

---

## Week 3 — Real-world messiness (this is what interview questions probe)

**Day 15 — Class imbalance**
Task: Check target class balance. Refit best model with `class_weight='balanced'`.
Done right if: minority-class recall improves; you can state the precision trade-off that came with it.

**Day 16 — ROC-AUC and PR curves**
Task: Plot ROC curve and Precision-Recall curve for your best model.
Done right if: you can state which curve matters more for *your* problem (imbalanced data → PR curve) and why.

**Day 17 — SMOTE**
Task: Apply SMOTE oversampling (imblearn), compare to `class_weight` approach from Day 15.
Done right if: you have a 3-row comparison table (baseline / class_weight / SMOTE) with a clear winner.

**Day 18 — Feature selection**
Task: Reduce to top 10 features using `feature_importances_` or `SelectKBest`, retrain, compare performance.
Done right if: reduced model is within 2% of full-feature performance.

**Day 19 — Save and reload the model**
Task: Save the model with `joblib`. Write a standalone `predict.py` script that loads it and returns a prediction for new input.
Done right if: running `python predict.py` from terminal (not the notebook) gives a correct prediction.

**Day 20 — Serve it**
Task: Wrap `predict.py` in a minimal FastAPI/Flask app with one `/predict` route.
Done right if: a `curl` or Postman request to `localhost` returns a JSON prediction.
Don't: don't add authentication, logging, or extra routes. One route only.

**Day 21 — Ship the repo**
Task: Push the full project to GitHub with a README (problem, data, approach, results).
Done right if: repo is public and a stranger could understand the project from the README in 2 minutes.

---

## Week 4 — MLOps basics + interview readiness

**Day 22 — Git hygiene**
Task: Create a feature branch, make a change, commit, merge back. Add a proper `.gitignore` (exclude data files, model files).
Done right if: clean merge, no data/model files tracked in git.

**Day 23 — Experiment tracking**
Task: Log 3 different model runs (params + metrics) either in MLflow or a simple CSV, side by side.
Done right if: you can compare 3 runs in one table/view without re-running code.

**Day 24 — Containerize it**
Task: Write a minimal Dockerfile for your `predict.py` API. Build and run it.
Done right if: `docker build` succeeds and `docker run` serves the API locally.

**Day 25 — Prove it wasn't luck**
Task: Pick a second small dataset. Solo, timed, no notebook reference: load → clean → model → evaluate.
Done right if: you finish a working pipeline in 45 minutes unaided. This is the real test of whether Week 1-3 stuck.

**Day 26 — Tell the story**
Task: Write your project as a STAR answer (Situation/Task/Action/Result).
Done right if: it's under 90 seconds spoken aloud, with zero unexplained jargon.

**Day 27 — Common questions, your own words**
Task: Write answers to: bias-variance tradeoff, overfitting, precision vs recall, when RF vs GB, handling imbalance.
Done right if: each answer is under 90 seconds spoken, and none of it is copy-pasted from an article.

**Day 28 — Minimal RAG exposure**
Task: Embed 5 short text docs, retrieve the most relevant one for a query via cosine similarity, pass it to an LLM API call.
Done right if: given a test query, the correct document is retrieved and used in the response.
Don't: don't try to build a "production RAG system" today. 5 docs, one retrieval, one call — that's it.

**Day 29 — Resume bullets**
Task: Rewrite 2 resume bullets describing this project with concrete numbers (dataset size, metric improvement).
Done right if: each bullet passes the "so what" test — impact is clear on one read.

**Day 30 — Full dry run**
Task: Record yourself (audio/video) answering 3 project questions + 2 technical questions back to back, no notes.
Done right if: you watch it back and can name your top 2 weak spots to keep drilling after Day 30.

---

## What this plan does NOT make you
It does not make you deep in deep learning, PyTorch, LLM fine-tuning, or MLOps at scale. It makes you someone who can say "I built this, here's how, here's why," and back it up under questioning — which is the actual bar you failed to clear before, not a breadth-of-topics bar.
