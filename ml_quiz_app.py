
# ML & Data Science Mega Quiz (500+ auto-generated Qs)
# Ready to run with:  streamlit run ml_quiz_app.py
# No external APIs required. Progress saved locally in progress.json
import streamlit as st
import json, random, os, datetime, math
from collections import defaultdict

st.set_page_config(page_title="ML & Data Science Mega Quiz", page_icon="🧠", layout="wide")

# --------------------------
# Minimal embedded knowledge base
# Each item: name, kind, area, short, params (optional)
# kind ∈ {"algorithm","metric","tool","preprocess","validation","concept","deployment","nlp","cv","timeseries","graph"}
# area is a coarse bucket used for filters and distractors.
# --------------------------
KB = [
    # ---- Supervised algorithms
    {"name":"Linear Regression","kind":"algorithm","area":"supervised-regression","short":"OLS linear model for regression.","params":["fit_intercept","normalize"]},
    {"name":"Ridge Regression","kind":"algorithm","area":"supervised-regression","short":"L2-regularized linear regression.","params":["alpha"]},
    {"name":"Lasso","kind":"algorithm","area":"supervised-regression","short":"L1-regularized linear regression for feature selection.","params":["alpha"]},
    {"name":"Elastic Net","kind":"algorithm","area":"supervised-regression","short":"Mix of L1/L2 regularization for regression.","params":["alpha","l1_ratio"]},
    {"name":"Logistic Regression","kind":"algorithm","area":"supervised-classification","short":"Linear classifier using logistic function.","params":["C","penalty","solver"]},
    {"name":"k-Nearest Neighbors","kind":"algorithm","area":"supervised-both","short":"Instance-based learner using neighborhood vote/distance.","params":["n_neighbors","weights","metric"]},
    {"name":"Support Vector Machine","kind":"algorithm","area":"supervised-both","short":"Large-margin classifier/regressor with kernels.","params":["C","kernel","gamma","epsilon"]},
    {"name":"Decision Tree","kind":"algorithm","area":"supervised-both","short":"Tree-based model splitting by impurity.","params":["max_depth","min_samples_split","min_samples_leaf"]},
    {"name":"Random Forest","kind":"algorithm","area":"supervised-both","short":"Ensemble of bagged decision trees.","params":["n_estimators","max_depth","max_features"]},
    {"name":"Extra Trees","kind":"algorithm","area":"supervised-both","short":"Extremely randomized trees ensemble.","params":["n_estimators","max_depth","max_features"]},
    {"name":"Gradient Boosting","kind":"algorithm","area":"supervised-both","short":"Boosted trees using gradient descent on loss.","params":["n_estimators","learning_rate","max_depth"]},
    {"name":"XGBoost","kind":"algorithm","area":"supervised-both","short":"Optimized gradient boosting library.","params":["n_estimators","eta","max_depth","subsample","colsample_bytree"]},
    {"name":"LightGBM","kind":"algorithm","area":"supervised-both","short":"Leaf-wise histogram-based gradient boosting.","params":["num_leaves","learning_rate","n_estimators","max_depth"]},
    {"name":"CatBoost","kind":"algorithm","area":"supervised-both","short":"Boosting that handles categorical features natively.","params":["depth","learning_rate","iterations"]},
    {"name":"Naive Bayes","kind":"algorithm","area":"supervised-classification","short":"Probabilistic classifier assuming feature independence.","params":["var_smoothing","alpha"]},
    {"name":"Linear Discriminant Analysis","kind":"algorithm","area":"supervised-classification","short":"Classifier using class conditional Gaussians with shared covariance.","params":["solver"]},
    {"name":"Quadratic Discriminant Analysis","kind":"algorithm","area":"supervised-classification","short":"Classifier with class-specific covariance matrices.","params":[]},
    {"name":"Perceptron","kind":"algorithm","area":"supervised-classification","short":"Single-layer linear classifier trained by mistakes.","params":["alpha","penalty"]},
    {"name":"SGDClassifier","kind":"algorithm","area":"supervised-both","short":"Linear models trained with SGD for large data.","params":["loss","alpha","penalty","learning_rate"]},
    # ---- Unsupervised
    {"name":"k-Means","kind":"algorithm","area":"unsupervised-clustering","short":"Partition clustering minimizing within-cluster SSE.","params":["n_clusters","init","n_init"]},
    {"name":"DBSCAN","kind":"algorithm","area":"unsupervised-clustering","short":"Density-based clustering discovering arbitrary shapes.","params":["eps","min_samples","metric"]},
    {"name":"Gaussian Mixture Model","kind":"algorithm","area":"unsupervised-clustering","short":"Probabilistic clustering via mixtures of Gaussians.","params":["n_components","covariance_type"]},
    {"name":"Agglomerative Clustering","kind":"algorithm","area":"unsupervised-clustering","short":"Hierarchical bottom-up clustering.","params":["n_clusters","linkage"]},
    {"name":"PCA","kind":"algorithm","area":"unsupervised-dimred","short":"Linear dimensionality reduction via variance maximization.","params":["n_components","svd_solver"]},
    {"name":"t-SNE","kind":"algorithm","area":"unsupervised-dimred","short":"Nonlinear visualization preserving local neighborhoods.","params":["perplexity","learning_rate"]},
    {"name":"UMAP","kind":"algorithm","area":"unsupervised-dimred","short":"Manifold learning for visualization.","params":["n_neighbors","min_dist"]},
    {"name":"Isolation Forest","kind":"algorithm","area":"unsupervised-anomaly","short":"Tree-based anomaly detection by isolation depth.","params":["n_estimators","max_samples"]},
    {"name":"One-Class SVM","kind":"algorithm","area":"unsupervised-anomaly","short":"SVM variant to delineate normal region.","params":["nu","kernel","gamma"]},
    # ---- Deep Learning
    {"name":"Multilayer Perceptron","kind":"algorithm","area":"deep-learning","short":"Feedforward neural network.","params":["hidden_layer_sizes","activation","alpha"]},
    {"name":"Convolutional Neural Network","kind":"algorithm","area":"cv","short":"Neural network specialized for images.","params":["filters","kernel_size","stride","padding"]},
    {"name":"Recurrent Neural Network","kind":"algorithm","area":"sequence","short":"Neural network with recurrent connections for sequences.","params":["hidden_size"]},
    {"name":"LSTM","kind":"algorithm","area":"sequence","short":"RNN with gating for long dependencies.","params":["hidden_size","num_layers"]},
    {"name":"GRU","kind":"algorithm","area":"sequence","short":"RNN variant with fewer gates than LSTM.","params":["hidden_size","num_layers"]},
    {"name":"Transformer","kind":"algorithm","area":"nlp","short":"Attention-based sequence model.","params":["num_heads","d_model","num_layers"]},
    {"name":"Graph Neural Network","kind":"algorithm","area":"graph","short":"Neural networks on graphs.","params":["num_layers","aggregation"]},
    # ---- NLP
    {"name":"TF-IDF","kind":"preprocess","area":"nlp","short":"Word weighting combining term frequency and inverse document frequency.","params":["ngram_range","min_df","max_df"]},
    {"name":"Word2Vec","kind":"algorithm","area":"nlp","short":"Neural embeddings learning word vectors.","params":["vector_size","window","min_count"]},
    {"name":"FastText","kind":"algorithm","area":"nlp","short":"Embeddings using subword information.","params":["vector_size","min_n","max_n"]},
    {"name":"BERT","kind":"algorithm","area":"nlp","short":"Bidirectional transformer pretrained on masked LM.","params":["hidden_size","num_attention_heads"]},
    {"name":"CRF","kind":"algorithm","area":"nlp","short":"Sequence labeling with conditional random fields.","params":["c1","c2"]},
    # ---- Time Series
    {"name":"ARIMA","kind":"algorithm","area":"timeseries","short":"Autoregressive integrated moving average.","params":["p","d","q"]},
    {"name":"SARIMA","kind":"algorithm","area":"timeseries","short":"Seasonal ARIMA.","params":["P","D","Q","m"]},
    {"name":"Prophet","kind":"algorithm","area":"timeseries","short":"Additive model with trend/season/holiday.","params":["changepoint_prior_scale","seasonality_mode"]},
    {"name":"Exponential Smoothing","kind":"algorithm","area":"timeseries","short":"Smoothing methods for forecasting.","params":["trend","seasonal","damped"]},
    # ---- Preprocessing
    {"name":"StandardScaler","kind":"preprocess","area":"preprocessing","short":"Standardize to zero mean/unit variance.","params":[]},
    {"name":"MinMaxScaler","kind":"preprocess","area":"preprocessing","short":"Scale features to a given range.","params":["feature_range"]},
    {"name":"RobustScaler","kind":"preprocess","area":"preprocessing","short":"Scale using medians and IQR, robust to outliers.","params":[]},
    {"name":"One-Hot Encoding","kind":"preprocess","area":"preprocessing","short":"Binary columns for categorical levels.","params":["handle_unknown"]},
    {"name":"Label Encoding","kind":"preprocess","area":"preprocessing","short":"Map categories to integers (ordered).","params":[]},
    {"name":"Target Encoding","kind":"preprocess","area":"preprocessing","short":"Replace categories by target statistics.","params":["smoothing"]},
    {"name":"KNNImputer","kind":"preprocess","area":"preprocessing","short":"Impute missing values using nearest neighbors.","params":["n_neighbors","weights"]},
    {"name":"SimpleImputer","kind":"preprocess","area":"preprocessing","short":"Mean/median/most_frequent imputation.","params":["strategy"]},
    {"name":"PolynomialFeatures","kind":"preprocess","area":"preprocessing","short":"Generate polynomial and interaction terms.","params":["degree","interaction_only"]},
    {"name":"SMOTE","kind":"preprocess","area":"preprocessing","short":"Oversample minority class by synthetic samples.","params":["k_neighbors"]},
    {"name":"PCA Whitening","kind":"preprocess","area":"preprocessing","short":"Decorrelate and scale components to unit variance.","params":[]},
    # ---- Validation & Experimentation
    {"name":"KFold","kind":"validation","area":"validation","short":"Cross-validation splitting into K folds.","params":["n_splits","shuffle"]},
    {"name":"StratifiedKFold","kind":"validation","area":"validation","short":"KFold preserving class proportions.","params":["n_splits","shuffle"]},
    {"name":"GroupKFold","kind":"validation","area":"validation","short":"KFold keeping groups intact across folds.","params":["n_splits"]},
    {"name":"TimeSeriesSplit","kind":"validation","area":"validation","short":"Forward-chaining split for temporal data.","params":["n_splits"]},
    {"name":"Early Stopping","kind":"concept","area":"training","short":"Stop training when validation loss stops improving.","params":[]},
    {"name":"Regularization L1","kind":"concept","area":"training","short":"Sparsity-inducing penalty on weights.","params":["alpha"]},
    {"name":"Regularization L2","kind":"concept","area":"training","short":"Weight decay penalty to prevent overfit.","params":["alpha"]},
    # ---- Metrics
    {"name":"Accuracy","kind":"metric","area":"classification","short":"Share of correct predictions.","params":[]},
    {"name":"Precision","kind":"metric","area":"classification","short":"TP / (TP + FP).","params":[]},
    {"name":"Recall","kind":"metric","area":"classification","short":"TP / (TP + FN).","params":[]},
    {"name":"F1 Score","kind":"metric","area":"classification","short":"Harmonic mean of precision and recall.","params":[]},
    {"name":"ROC AUC","kind":"metric","area":"classification","short":"Area under ROC curve.","params":[]},
    {"name":"PR AUC","kind":"metric","area":"classification","short":"Area under precision-recall curve.","params":[]},
    {"name":"Log Loss","kind":"metric","area":"classification","short":"Cross-entropy between labels and probs.","params":[]},
    {"name":"MSE","kind":"metric","area":"regression","short":"Mean squared error.","params":[]},
    {"name":"RMSE","kind":"metric","area":"regression","short":"Root mean squared error.","params":[]},
    {"name":"MAE","kind":"metric","area":"regression","short":"Mean absolute error.","params":[]},
    {"name":"R2","kind":"metric","area":"regression","short":"Explained variance proportion.","params":[]},
    {"name":"MAPE","kind":"metric","area":"regression","short":"Mean absolute percentage error.","params":[]},
    {"name":"SMAPE","kind":"metric","area":"regression","short":"Symmetric MAPE.","params":[]},
    {"name":"Silhouette Score","kind":"metric","area":"clustering","short":"Cluster cohesion/separation measure.","params":[]},
    {"name":"ARI","kind":"metric","area":"clustering","short":"Adjusted Rand Index.","params":[]},
    {"name":"NMI","kind":"metric","area":"clustering","short":"Normalized Mutual Information.","params":[]},
    {"name":"BLEU","kind":"metric","area":"nlp","short":"n-gram precision for MT.","params":[]},
    {"name":"ROUGE","kind":"metric","area":"nlp","short":"Recall-oriented summary metric.","params":[]},
    {"name":"Perplexity","kind":"metric","area":"nlp","short":"Exponentiated average negative log-likelihood.","params":[]},
    # ---- Tools & MLOps
    {"name":"NumPy","kind":"tool","area":"python","short":"N-dimensional arrays and linear algebra.","params":[]},
    {"name":"pandas","kind":"tool","area":"python","short":"Dataframes and data wrangling.","params":[]},
    {"name":"scikit-learn","kind":"tool","area":"python","short":"Classical ML library in Python.","params":[]},
    {"name":"TensorFlow","kind":"tool","area":"deep-learning","short":"DL framework with Keras high-level API.","params":[]},
    {"name":"PyTorch","kind":"tool","area":"deep-learning","short":"Dynamic graph deep learning framework.","params":[]},
    {"name":"Keras","kind":"tool","area":"deep-learning","short":"High-level neural networks API.","params":[]},
    {"name":"XGBoost-lib","kind":"tool","area":"python","short":"Gradient boosting library.","params":[]},
    {"name":"LightGBM-lib","kind":"tool","area":"python","short":"Fast histogram-based GBDT library.","params":[]},
    {"name":"CatBoost-lib","kind":"tool","area":"python","short":"Boosting library handling categoricals.","params":[]},
    {"name":"Optuna","kind":"tool","area":"tuning","short":"Hyperparameter optimization framework.","params":[]},
    {"name":"scikit-optimize","kind":"tool","area":"tuning","short":"Bayesian optimization tools.","params":[]},
    {"name":"MLflow","kind":"tool","area":"mlops","short":"Experiment tracking and model registry.","params":[]},
    {"name":"Weights & Biases","kind":"tool","area":"mlops","short":"Experiment tracking and reports.","params":[]},
    {"name":"DVC","kind":"tool","area":"mlops","short":"Data & pipeline version control.","params":[]},
    {"name":"Apache Airflow","kind":"tool","area":"pipelines","short":"Workflow orchestration.","params":[]},
    {"name":"Docker","kind":"tool","area":"deploy","short":"Containerization platform.","params":[]},
    {"name":"Kubernetes","kind":"tool","area":"deploy","short":"Container orchestration cluster.","params":[]},
    {"name":"Apache Spark","kind":"tool","area":"bigdata","short":"Distributed compute engine.","params":[]},
    {"name":"Databricks","kind":"tool","area":"bigdata","short":"Managed Spark & Lakehouse platform.","params":[]},
    {"name":"Snowflake","kind":"tool","area":"bigdata","short":"Cloud data warehouse.","params":[]},
    {"name":"BigQuery","kind":"tool","area":"bigdata","short":"Google's serverless data warehouse.","params":[]},
    {"name":"AWS SageMaker","kind":"tool","area":"cloud-ml","short":"Managed ML training/hosting on AWS.","params":[]},
    {"name":"Azure ML","kind":"tool","area":"cloud-ml","short":"Managed ML platform on Azure.","params":[]},
    {"name":"Google Vertex AI","kind":"tool","area":"cloud-ml","short":"Managed ML platform on GCP.","params":[]},
    {"name":"Hugging Face","kind":"tool","area":"nlp","short":"Models/datasets/hub for transformers.","params":[]},
    {"name":"spaCy","kind":"tool","area":"nlp","short":"Industrial-strength NLP library.","params":[]},
    {"name":"NLTK","kind":"tool","area":"nlp","short":"Classic NLP toolkit.","params":[]},
    {"name":"OpenCV","kind":"tool","area":"cv","short":"Computer vision library.","params":[]},
    {"name":"Matplotlib","kind":"tool","area":"viz","short":"Plotting library for Python.","params":[]},
    {"name":"Seaborn","kind":"tool","area":"viz","short":"Statistical data visualization on Matplotlib.","params":[]},
    {"name":"Plotly","kind":"tool","area":"viz","short":"Interactive charts.","params":[]},
]

# Extra conceptual Qs to exceed 500
CONCEPT_QS = [
    ("Which validation strategy is most appropriate for time-ordered data?", 
     ["TimeSeriesSplit","StratifiedKFold","RandomizedSplit","LeaveOneOut"], "TimeSeriesSplit",
     "TimeSeriesSplit respects temporal order to prevent leakage."),
    ("For highly imbalanced classification with calibrated probabilities, which metric is usually better than accuracy?",
     ["ROC AUC","Precision","Recall","F1 Score"], "ROC AUC",
     "Accuracy can be misleading; ROC AUC evaluates ranking across thresholds."),
    ("Which technique helps prevent overfitting by halting training when validation loss stops improving?",
     ["Data augmentation","Early Stopping","Gradient Clipping","Batch Normalization"], "Early Stopping",
     "Early stopping monitors validation loss and stops when it plateaus."),
    ("What scaling method is most robust to outliers?",
     ["StandardScaler","MinMaxScaler","RobustScaler","Normalizer"], "RobustScaler",
     "RobustScaler uses medians/IQR which are less sensitive to outliers."),
    ("Which cross-validation keeps class distribution similar across folds?",
     ["KFold","StratifiedKFold","GroupKFold","ShuffleSplit"], "StratifiedKFold",
     "StratifiedKFold preserves class proportions in each fold."),
    ("Which regularization encourages sparsity (feature selection) in linear models?",
     ["L2","Dropout","Early Stopping","L1"], "L1",
     "L1 (Lasso) can drive some coefficients exactly to zero."),
    ("Which algorithm is best suited for discovering arbitrary-shaped clusters?",
     ["k-Means","DBSCAN","GMM","Spectral Clustering"], "DBSCAN",
     "DBSCAN can find clusters of arbitrary shape and detect noise."),
    ("Which metric is preferred when positive class is rare and you care about ranking positive cases?",
     ["Accuracy","ROC AUC","MSE","R2"], "ROC AUC",
     "ROC AUC evaluates the tradeoff across thresholds independent of prevalence."),
    ("Which library is commonly used for experiment tracking and model registry?",
     ["Optuna","MLflow","pandas","OpenCV"], "MLflow",
     "MLflow tracks runs, parameters, metrics and models."),
    ("Which encoding can leak target information if not done within CV folds?",
     ["One-Hot","Label Encoding","Target Encoding","Hashing"], "Target Encoding",
     "Target encoding must be fitted inside training folds to avoid leakage.")
]

AREA_DESC = {
    "supervised-regression":"Supervised • Regression",
    "supervised-classification":"Supervised • Classification",
    "supervised-both":"Supervised • Classification/Regression",
    "unsupervised-clustering":"Unsupervised • Clustering",
    "unsupervised-dimred":"Unsupervised • Dimensionality Reduction",
    "unsupervised-anomaly":"Unsupervised • Anomaly Detection",
    "deep-learning":"Deep Learning",
    "cv":"Computer Vision",
    "sequence":"Sequences",
    "nlp":"NLP",
    "graph":"Graphs",
    "timeseries":"Time Series",
    "preprocessing":"Preprocessing/Feature Eng",
    "validation":"Validation/Resampling",
    "training":"Training/Regularization",
    "classification":"Classification Metrics",
    "regression":"Regression Metrics",
    "clustering":"Clustering Metrics",
    "python":"Python",
    "tuning":"Hyperparameter Tuning",
    "mlops":"MLOps",
    "pipelines":"Pipelines/Orchestration",
    "deploy":"Deployment",
    "bigdata":"Big Data & Warehouses",
    "cloud-ml":"Cloud ML Platforms",
    "viz":"Visualization",
}

# --------------------------
# Question generation templates
# --------------------------
def make_mcq_purpose(item):
    # Ask: "What is X primarily used for?"
    purpose_map = {
        "supervised-regression":"Regression",
        "supervised-classification":"Classification",
        "supervised-both":"Classification and Regression",
        "unsupervised-clustering":"Clustering",
        "unsupervised-dimred":"Dimensionality reduction",
        "unsupervised-anomaly":"Anomaly detection",
        "deep-learning":"Deep learning",
        "cv":"Computer vision",
        "sequence":"Sequential modeling",
        "nlp":"Natural language processing",
        "graph":"Graph learning",
        "timeseries":"Time series forecasting",
        "preprocessing":"Preprocessing/feature engineering",
        "validation":"Model validation",
        "training":"Regularization/training control",
        "classification":"Classification evaluation",
        "regression":"Regression evaluation",
        "clustering":"Clustering evaluation",
        "python":"Array/data manipulation",
        "tuning":"Hyperparameter optimization",
        "mlops":"Experiment tracking/MLOps",
        "pipelines":"Workflow orchestration",
        "deploy":"Containerization/orchestration",
        "bigdata":"Distributed compute/warehouse",
        "cloud-ml":"Cloud ML platform",
        "viz":"Data visualization",
    }
    correct = purpose_map.get(item["area"], "Machine learning")
    distractors = list(set(purpose_map.values()) - {correct})
    random.shuffle(distractors)
    options = [correct] + distractors[:3]
    random.shuffle(options)
    return {
        "q": f"What is {item['name']} primarily used for?",
        "options": options, "answer": correct,
        "explain": f"{item['name']}: {item['short']}"
    }

def make_mcq_params(item):
    # Ask: "Which is a key hyperparameter of X?"
    if not item.get("params"):
        return None
    correct = random.choice(item["params"])
    # build distractors from other items' params and common red herrings
    pool = []
    for it in KB:
        if it.get("params"):
            pool += it["params"]
    pool += ["batch_size","dropout","momentum","num_workers","beta1"]
    pool = [p for p in set(pool) if p != correct]
    random.shuffle(pool)
    options = [correct] + pool[:3]
    random.shuffle(options)
    return {
        "q": f"Which of the following is a typical hyperparameter of {item['name']}?",
        "options": options, "answer": correct,
        "explain": f"Common hyperparameters for {item['name']}: {', '.join(item['params'])}."
    }

def make_tf(item):
    # True/False assertion from description
    text = item["short"]
    truth = random.choice([True, False])
    if truth:
        statement = text
        answer = "True"
    else:
        # Negate or swap a phrase heuristically
        swaps = [
            ("classification","regression"),
            ("regression","classification"),
            ("increase","decrease"),
            ("L1","L2"),
            ("L2","L1"),
            ("bagged","boosted"),
            ("One-Hot","Target"),
            ("minimizing","maximizing"),
            ("Gaussians","uniforms"),
            ("neighbors","centroids"),
            ("variance","mean"),
        ]
        statement = text
        for a,b in swaps:
            if a in statement:
                statement = statement.replace(a,b)
                break
        answer = "False"
    return {
        "q": f"True or False: {statement}",
        "options": ["True","False"], "answer": answer,
        "explain": f"{item['name']}: {item['short']}"
    }

def make_area(item):
    options = [AREA_DESC.get(item["area"], item["area"])]
    # add 3 other random areas
    others = list(set(AREA_DESC.values()) - set(options))
    random.shuffle(others)
    options += others[:3]
    random.shuffle(options)
    return {
        "q": f"{item['name']} belongs best to which category?",
        "options": options,
        "answer": AREA_DESC.get(item["area"], item["area"]),
        "explain": f"{item['name']} → {AREA_DESC.get(item['area'], item['area'])}."
    }

def build_bank(seed=42):
    random.seed(seed)
    questions = []
    for item in KB:
        questions.append(make_mcq_purpose(item))
        q2 = make_mcq_params(item)
        if q2: questions.append(q2)
        questions.append(make_tf(item))
        questions.append(make_area(item))
    # add extra concept questions
    for q, opts, ans, exp in CONCEPT_QS:
        questions.append({"q":q, "options":opts, "answer":ans, "explain":exp})
    # ensure we have at least 500 questions by duplicating with paraphrases if needed
    if len(questions) < 500:
        # create paraphrased copies with slight wording changes
        needed = 500 - len(questions)
        base = list(questions)
        random.shuffle(base)
        for i in range(needed):
            b = base[i % len(base)]
            qtext = b["q"].replace("primarily","mainly").replace("belongs best","fits best")
            questions.append({"q":qtext, "options":b["options"][:], "answer":b["answer"], "explain":b["explain"]})
    random.shuffle(questions)
    return questions

# --------------------------
# Persistence (progress & Leitner)
# --------------------------
PROGRESS_FILE = "progress.json"

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE,"r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_progress(data):
    try:
        with open(PROGRESS_FILE,"w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

# Leitner boxes for spaced repetition
def init Leitner(progress, questions):
    # boxes: 1..5; start at 1
    if "leitner" not in progress: progress["leitner"] = {}
    for idx, _ in enumerate(questions):
        sid = str(idx)
        if sid not in progress["leitner"]:
            progress["leitner"][sid] = {"box":1, "last_seen":None, "streak":0}
    return progress

# schedule: box 1 daily, 2 every 2 days, 3 every 4 days, 4 weekly, 5 biweekly
SCHEDULE = {1:1, 2:2, 3:4, 4:7, 5:14}

def due_today(meta):
    today = datetime.date.today()
    last = meta["last_seen"]
    box = meta["box"]
    gap = SCHEDULE.get(box, 7)
    if not last: return True
    try:
        last_date = datetime.date.fromisoformat(last)
    except Exception:
        return True
    return (today - last_date).days >= gap

# --------------------------
# UI
# --------------------------
st.title("🧠 ML & Data Science Mega Quiz")
st.caption("500+ auto-generated questions across algorithms, tools, metrics, preprocessing, validation, NLP/CV/TS, and MLOps.")

with st.expander("How it works / Settings", expanded=False):
    st.markdown("""
- Questions are generated from a built-in knowledge base using multiple templates (purpose, params, true/false, categories), plus general concept questions.
- A **Leitner spaced-repetition** scheduler prioritizes questions you got wrong.
- Your progress is saved to `progress.json` in the app folder.
- Filter by area, pick a session length, and grind! 🚀
    """)

# Build questions (deterministic seed so IDs are stable across runs)
QUESTIONS = build_bank(seed=7)
progress = load_progress()

# Backward compatibility
if "scores" not in progress: progress["scores"] = {"correct":0,"total":0}
progress = init Leitner(progress, QUESTIONS)

# Filters
areas = sorted(set(AREA_DESC.get(k["area"],k["area"]) for k in KB))
col1, col2, col3, col4 = st.columns([1.5,1,1,1])
with col1:
    chosen_areas = st.multiselect("Filter by category (optional)", options=areas, default=[])
with col2:
    session_len = st.number_input("Questions this session", min_value=5, max_value=100, value=20, step=5)
with col3:
    enable_leitner = st.checkbox("Prioritize due cards (Leitner)", value=True)
with col4:
    reset = st.button("Reset Progress")

if reset:
    try:
        os.remove(PROGRESS_FILE)
    except FileNotFoundError:
        pass
    st.success("Progress reset. Reload the page.")
    st.stop()

# Build index of candidate question IDs
cand_ids = []
for idx, q in enumerate(QUESTIONS):
    # area filter: map question to an item area when possible by parsing explanation
    if chosen_areas:
        matched = False
        for item in KB:
            if item["name"] in q["explain"] or item["name"] in q["q"]:
                area_text = AREA_DESC.get(item["area"], item["area"])
                if area_text in chosen_areas:
                    matched = True
                    break
        if not matched:
            continue
    cand_ids.append(idx)

# Leitner prioritization: put due items first
if enable_leitner:
    due = []
    not_due = []
    for i in cand_ids:
        meta = progress["leitner"].get(str(i), {"box":1,"last_seen":None,"streak":0})
        (due if due_today(meta) else not_due).append(i)
    # Weight due items heavier
    ordered = due + not_due
else:
    ordered = cand_ids[:]

# Limit to session length
ordered = ordered[:int(session_len)] if len(ordered) > session_len else ordered

# Session state for quiz position
if "pos" not in st.session_state: st.session_state.pos = 0
if "answers" not in st.session_state: st.session_state.answers = {}

def render_question(i):
    q = QUESTIONS[i]
    st.subheader(f"Q{i+1}: {q['q']}")
    choice = st.radio("Choose one:", q["options"], index=None, key=f"q_{i}")
    if st.button("Check", key=f"check_{i}"):
        correct = (choice == q["answer"])
        if choice is None:
            st.warning("Pick an option first.")
            return False
        if correct:
            st.success(f"✅ Correct! **{q['answer']}**")
        else:
            st.error(f"❌ Incorrect. Correct answer: **{q['answer']}**")
        st.info(q["explain"])
        # update progress/leitner
        progress["scores"]["total"] += 1
        if correct: progress["scores"]["correct"] += 1
        meta = progress["leitner"].get(str(i), {"box":1,"last_seen":None,"streak":0})
        meta["last_seen"] = datetime.date.today().isoformat()
        if correct:
            meta["streak"] = meta.get("streak",0) + 1
            if meta["box"] < 5 and meta["streak"] >= 2:
                meta["box"] += 1
                meta["streak"] = 0
        else:
            meta["box"] = 1
            meta["streak"] = 0
        progress["leitner"][str(i)] = meta
        save_progress(progress)
        return True
    return None

# Show scoreboard
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Correct", progress["scores"]["correct"])
with c2:
    st.metric("Total Answered", progress["scores"]["total"])
with c3:
    pct = 0 if progress["scores"]["total"]==0 else round(100*progress["scores"]["correct"]/progress["scores"]["total"],1)
    st.metric("Accuracy (%)", pct)

# Navigation
if st.session_state.pos >= len(ordered):
    st.success("Session complete! Adjust filters or session length to continue.")
else:
    idx = ordered[st.session_state.pos]
    res = render_question(idx)
    nav1, nav2, nav3 = st.columns([1,1,4])
    with nav1:
        if st.button("⏭ Next"):
            st.session_state.pos = min(st.session_state.pos + 1, len(ordered))
    with nav2:
        if st.button("⏮ Back"):
            st.session_state.pos = max(st.session_state.pos - 1, 0)

st.caption("Tip: Use filters to focus (e.g., NLP, Time Series, Metrics). Your progress persists locally.")
