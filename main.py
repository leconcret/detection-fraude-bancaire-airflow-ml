# API FastAPI - Detection de Fraude Bancaire (temps reel)
# Utilise le modele mis a jour automatiquement par le pipeline Airflow

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from sqlalchemy import create_engine

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DB_CONNECTION = "postgresql://postgres:root@localhost:5432/banque_fraude"

app = FastAPI(
    title="API de Detection de Fraude Bancaire",
    description="Detection en temps reel, connectee au pipeline Airflow qui reentraine le modele automatiquement.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = create_engine(DB_CONNECTION)

etat_modele = {"model": None, "scaler": None, "colonnes": None, "metadata": None}


def charger_modele():
    etat_modele["model"] = joblib.load(os.path.join(MODELS_DIR, "FINAL_MODEL_fraude.pkl"))
    etat_modele["scaler"] = joblib.load(os.path.join(MODELS_DIR, "scaler_fraude.pkl"))
    etat_modele["colonnes"] = joblib.load(os.path.join(MODELS_DIR, "colonnes_modele_fraude.pkl"))
    meta_path = os.path.join(MODELS_DIR, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            etat_modele["metadata"] = json.load(f)
    else:
        etat_modele["metadata"] = {}


charger_modele()

PAYS_CONNUS = ["France", "Nigeria", "Russie", "Chine", "Bresil", "Roumanie", "Ukraine"]
TYPES_CONNUS = ["Depot", "Retrait", "Virement", "Paiement"]
SEGMENTS_CONNUS = ["Particulier", "Professionnel"]


class NouvelleTransaction(BaseModel):
    compte_id: int = Field(..., description="Identifiant du compte", example=440)
    montant: float = Field(..., description="Montant de la transaction", example=1500.0)
    type_transaction: str = Field(..., description="Retrait / Depot / Virement / Paiement", example="Virement")
    pays: str = Field(..., description="Pays de la transaction", example="France")


@app.get("/")
def root():
    return {
        "message": "API de Detection de Fraude - operationnelle",
        "documentation": "/docs",
        "endpoints": ["/predict", "/model-info", "/reload-model", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "auc": etat_modele["metadata"].get("auc"),
        "horodatage_dernier_entrainement": etat_modele["metadata"].get("horodatage"),
        "total_lignes_dernier_entrainement": etat_modele["metadata"].get("total_lignes_dernier_entrainement"),
        "nb_variables_modele": len(etat_modele["colonnes"]) if etat_modele["colonnes"] else 0,
    }


@app.post("/reload-model")
def reload_model():
    try:
        charger_modele()
        return {"message": "Modele recharge avec succes.", "metadata": etat_modele["metadata"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du rechargement : {str(e)}")


@app.post("/predict")
def predict(transaction: NouvelleTransaction):
    try:
        with engine.connect() as conn:
            info_compte = pd.read_sql(
                f"""SELECT c.segment FROM comptes cp
                    JOIN clients c ON cp.client_id = c.client_id
                    WHERE cp.compte_id = {transaction.compte_id}""",
                conn,
            )
            if info_compte.empty:
                raise HTTPException(status_code=404, detail=f"Compte {transaction.compte_id} introuvable.")
            segment = info_compte["segment"].iloc[0]

            hist = pd.read_sql(
                f"SELECT AVG(montant) as moyenne FROM transactions WHERE compte_id = {transaction.compte_id}",
                conn,
            )
            montant_moyen_historique = hist["moyenne"].iloc[0]
            if pd.isna(montant_moyen_historique):
                montant_moyen_historique = transaction.montant

            recent = pd.read_sql(
                f"""SELECT COUNT(*) as n FROM transactions
                    WHERE compte_id = {transaction.compte_id}
                    AND date_transaction >= NOW() - INTERVAL '24 hours'""",
                conn,
            )
            nb_transactions_24h = int(recent["n"].iloc[0]) + 1

            deja_pays = pd.read_sql(
                f"""SELECT COUNT(*) as n FROM transactions
                    WHERE compte_id = {transaction.compte_id} AND pays = '{transaction.pays}'""",
                conn,
            )
            premiere_fois_ce_pays = 1 if deja_pays["n"].iloc[0] == 0 else 0

        heure_transaction = datetime.now().hour

        ligne = {
            "montant": transaction.montant,
            "montant_moyen_historique": float(montant_moyen_historique),
            "nb_transactions_24h": nb_transactions_24h,
            "premiere_fois_ce_pays": premiere_fois_ce_pays,
            "heure_transaction": heure_transaction,
        }
        for t in TYPES_CONNUS:
            ligne[f"type_transaction_{t}"] = 1 if transaction.type_transaction == t else 0
        for p in PAYS_CONNUS:
            ligne[f"pays_{p}"] = 1 if transaction.pays == p else 0
        for s in SEGMENTS_CONNUS:
            ligne[f"segment_{s}"] = 1 if segment == s else 0

        df_ligne = pd.DataFrame([ligne])
        df_ligne = df_ligne.reindex(columns=etat_modele["colonnes"], fill_value=0)

        scaled = etat_modele["scaler"].transform(df_ligne)
        pred = int(etat_modele["model"].predict(scaled)[0])
        proba = etat_modele["model"].predict_proba(scaled)[0]
        confiance = float(np.max(proba) * 100)

        return {
            "compte_id": transaction.compte_id,
            "est_fraude": bool(pred),
            "label": "Fraude" if pred == 1 else "Normal",
            "confiance_pourcentage": round(confiance, 2),
            "probabilite_fraude": round(float(proba[1]) * 100, 2),
            "features_calculees": ligne,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de prediction : {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)