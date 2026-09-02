from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from faker import Faker
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Connexion : host.docker.internal au lieu de localhost, car Airflow tourne dans Docker
DB_CONNECTION = "postgresql://postgres:root@host.docker.internal:5432/banque_fraude"


def generer_nouvelles_transactions(**kwargs):
    """Simule l'arrivee de nouvelles transactions bancaires (comme un vrai flux quotidien)."""
    engine = create_engine(DB_CONNECTION)
    fake = Faker("fr_FR")

    with engine.connect() as conn:
        comptes_ids = pd.read_sql("SELECT compte_id FROM comptes", conn)["compte_id"].values

    N_NOUVELLES = 500
    TAUX_FRAUDE = 0.03
    PAYS_NATIONAL = "France"
    PAYS_ETRANGERS = ["Nigeria", "Russie", "Chine", "Bresil", "Roumanie", "Ukraine"]
    TYPES_TRANSACTION = ["Retrait", "Depot", "Virement", "Paiement"]

    n_fraudes = int(N_NOUVELLES * TAUX_FRAUDE)
    indices_fraude = set(np.random.choice(N_NOUVELLES, n_fraudes, replace=False))

    transactions = []
    for i in range(N_NOUVELLES):
        compte_id = np.random.choice(comptes_ids)
        est_fraude = i in indices_fraude

        if est_fraude:
            montant = round(np.random.lognormal(mean=6.5, sigma=1.2), 2)
            pays = np.random.choice(PAYS_ETRANGERS + [PAYS_NATIONAL], p=[0.14]*6 + [0.16])
            type_transaction = np.random.choice(["Retrait", "Virement", "Paiement"], p=[0.3, 0.5, 0.2])
        else:
            montant = round(np.random.lognormal(mean=4.0, sigma=1.0), 2)
            pays = PAYS_NATIONAL
            type_transaction = np.random.choice(TYPES_TRANSACTION, p=[0.25, 0.2, 0.25, 0.3])

        transactions.append({
            "compte_id": int(compte_id),
            "montant": montant,
            "type_transaction": type_transaction,
            "date_transaction": datetime.now(),
            "pays": pays,
            "est_fraude": est_fraude,
        })

    df_nouvelles = pd.DataFrame(transactions)
    df_nouvelles.to_sql("transactions", engine, if_exists="append", index=False)
    print(f"{len(df_nouvelles)} nouvelles transactions inserees, dont {df_nouvelles['est_fraude'].sum()} fraudes.")


def verifier_pipeline(**kwargs):
    """Verifie que la vue de features est bien a jour apres l'insertion."""
    engine = create_engine(DB_CONNECTION)
    with engine.connect() as conn:
        total = pd.read_sql("SELECT COUNT(*) as n FROM transactions", conn)["n"][0]
        total_vue = pd.read_sql("SELECT COUNT(*) as n FROM vue_features_transactions", conn)["n"][0]
    print(f"Table transactions : {total} lignes")
    print(f"Vue features       : {total_vue} lignes")
    assert total == total_vue, "Incoherence detectee entre la table et la vue !"
    print("Pipeline verifie avec succes : table et vue sont synchronisees.")

def reentrainer_modele_si_necessaire(**kwargs):
    """Reentraine le modele seulement si assez de nouvelles transactions sont arrivees."""
    MODELS_DIR = "/opt/airflow/models"
    META_PATH = os.path.join(MODELS_DIR, "metadata.json")
    SEUIL_NOUVELLES_LIGNES = 500

    engine = create_engine(DB_CONNECTION)
    with engine.connect() as conn:
        total_actuel = pd.read_sql("SELECT COUNT(*) as n FROM transactions", conn)["n"][0]

    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
        dernier_total = meta.get("total_lignes_dernier_entrainement", 0)
    else:
        dernier_total = 0

    print(f"Total actuel : {total_actuel} | Dernier entrainement : {dernier_total} | "
          f"Nouvelles lignes : {total_actuel - dernier_total}")

    if (total_actuel - dernier_total) < SEUIL_NOUVELLES_LIGNES:
        print("Pas assez de nouvelles donnees, reentrainement ignore.")
        return

    print("Seuil atteint, reentrainement en cours...")
    df = pd.read_sql("SELECT * FROM vue_features_transactions", engine)
    df["montant_moyen_historique"] = df["montant_moyen_historique"].fillna(df["montant"])
    df_encode = pd.get_dummies(df, columns=["type_transaction", "pays", "segment"], drop_first=False)
    X = df_encode.drop(columns=["transaction_id", "compte_id", "date_transaction", "est_fraude"])
    y = df_encode["est_fraude"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    y_proba = rf.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"Nouveau modele entraine - AUC : {round(auc, 4)}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    horodatage = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    joblib.dump(rf, f"{MODELS_DIR}/model_{horodatage}.pkl")
    joblib.dump(rf, f"{MODELS_DIR}/FINAL_MODEL_fraude.pkl")
    joblib.dump(scaler, f"{MODELS_DIR}/scaler_fraude.pkl")
    joblib.dump(list(X.columns), f"{MODELS_DIR}/colonnes_modele_fraude.pkl")

    with open(META_PATH, "w") as f:
        json.dump({
            "total_lignes_dernier_entrainement": int(total_actuel),
            "horodatage": horodatage,
            "auc": float(auc),
        }, f, indent=2)

    print("Modele et metadata sauvegardes avec succes.")

with DAG(
    dag_id="pipeline_fraude_bancaire",
    description="Ingestion quotidienne de transactions + verification du pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["fraude", "banque"],
) as dag:

    tache_generation = PythonOperator(
        task_id="generer_transactions",
        python_callable=generer_nouvelles_transactions,
    )

    tache_verification = PythonOperator(
        task_id="verifier_pipeline",
        python_callable=verifier_pipeline,
    )

    tache_reentrainement = PythonOperator(
        task_id="reentrainer_modele_si_necessaire",
        python_callable=reentrainer_modele_si_necessaire,
    )

    tache_generation >> tache_verification >> tache_reentrainement