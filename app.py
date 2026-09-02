# Interface de Supervision - Detection de Fraude Bancaire (temps reel)
# Se connecte a l'API FastAPI, dont le modele est mis a jour automatiquement par Airflow

import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import os
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

st.set_page_config(page_title="Detection de Fraude - Supervision", page_icon="🏦", layout="wide")

API_URL = "http://127.0.0.1:8010"

if "journal_alertes" not in st.session_state:
    st.session_state.journal_alertes = []


def ajouter_alerte(compte_id, label, confiance):
    st.session_state.journal_alertes.insert(0, {
        "Horodatage": datetime.now().strftime("%H:%M:%S"),
        "Compte": compte_id,
        "Statut": label,
        "Confiance (%)": confiance,
    })


st.title("🏦 Supervision Temps Reel — Detection de Fraude Bancaire")
st.caption("Modele mis a jour automatiquement par le pipeline Airflow (reentrainement quotidien)")

try:
    reponse_health = requests.get(f"{API_URL}/health", timeout=3)
    api_ok = reponse_health.status_code == 200
except Exception:
    api_ok = False

if not api_ok:
    st.error(f"Impossible de contacter l'API sur {API_URL}. Verifie que `uvicorn main:app --port 8010` est bien lance.")
    st.stop()

onglet_predict, onglet_dashboard, onglet_journal = st.tabs(
    ["🔎 Prediction en temps reel", "📊 Etat du modele", "📋 Journal d'alertes"]
)

with onglet_predict:
    st.subheader("Analyser une nouvelle transaction")
    st.caption("Les variables comportementales (historique, recence, pays deja visite) sont calculees en direct depuis la base de donnees.")

    with st.form("formulaire_prediction"):
        col1, col2 = st.columns(2)
        with col1:
            compte_id = st.number_input("Identifiant du compte", min_value=1, value=440, step=1)
            montant = st.number_input("Montant de la transaction (€)", min_value=0.0, value=1500.0, step=10.0)
        with col2:
            type_transaction = st.selectbox("Type de transaction", ["Depot", "Retrait", "Virement", "Paiement"])
            pays = st.selectbox("Pays", ["France", "Nigeria", "Russie", "Chine", "Bresil", "Roumanie", "Ukraine"])

        soumis = st.form_submit_button("🔍 Analyser la transaction", use_container_width=True)

    if soumis:
        try:
            reponse = requests.post(f"{API_URL}/predict", json={
                "compte_id": int(compte_id),
                "montant": float(montant),
                "type_transaction": type_transaction,
                "pays": pays,
            }, timeout=10)

            if reponse.status_code == 404:
                st.error(f"Compte {compte_id} introuvable dans la base de donnees.")
            elif reponse.status_code != 200:
                st.error(f"Erreur API : {reponse.text}")
            else:
                data = reponse.json()
                couleur = "#e0673f" if data["est_fraude"] else "#2fbf8f"
                st.markdown(
                    f"""<div style="padding:20px;border-radius:8px;border:2px solid {couleur};background-color:{couleur}22;">
                    <h3 style="color:{couleur};margin:0;">Verdict : {data['label']}</h3>
                    <p style="margin:4px 0 0;">Confiance : <b>{data['confiance_pourcentage']}%</b> — 
                    Probabilite de fraude : <b>{data['probabilite_fraude']}%</b></p>
                    </div>""",
                    unsafe_allow_html=True,
                )

                with st.expander("Voir le detail des variables comportementales calculees"):
                    st.json(data["features_calculees"])

                ajouter_alerte(compte_id, data["label"], data["confiance_pourcentage"])
        except Exception as e:
            st.error(f"Erreur de connexion a l'API : {e}")

with onglet_dashboard:
    st.subheader("Modele actuellement en service")

    try:
        info = requests.get(f"{API_URL}/model-info", timeout=3).json()
        c1, c2, c3 = st.columns(3)
        c1.metric("AUC du modele", f"{info['auc']*100:.2f}%" if info.get("auc") else "N/A")
        c2.metric("Nombre de variables", info.get("nb_variables_modele", "N/A"))
        c3.metric("Lignes au dernier entrainement", info.get("total_lignes_dernier_entrainement", "N/A"))
        st.caption(f"Dernier entrainement : {info.get('horodatage_dernier_entrainement', 'inconnu')}")
    except Exception as e:
        st.warning(f"Impossible de recuperer les infos du modele : {e}")

    st.divider()
    st.subheader("Recharger le modele")
    st.caption("A utiliser apres un reentrainement automatique par Airflow, pour charger la derniere version sans redemarrer l'API.")

    if st.button("🔄 Recharger le modele depuis le disque", use_container_width=True):
        try:
            reponse = requests.post(f"{API_URL}/reload-model", timeout=10)
            if reponse.status_code == 200:
                nouvelles_infos = reponse.json()["metadata"]
                st.success(f"Modele recharge avec succes ! Entraine le {nouvelles_infos.get('horodatage')} "
                           f"(AUC : {nouvelles_infos.get('auc', 0)*100:.2f}%)")
            else:
                st.error(f"Erreur : {reponse.text}")
        except Exception as e:
            st.error(f"Erreur de connexion : {e}")

with onglet_journal:
    st.subheader("Journal des analyses de la session")

    if not st.session_state.journal_alertes:
        st.info("Aucune analyse effectuee pour l'instant.")
    else:
        df_journal = pd.DataFrame(st.session_state.journal_alertes)

        def colorer_statut(val):
            return "background-color: #e0673f33" if val == "Fraude" else "background-color: #2fbf8f33"

        st.dataframe(df_journal.style.applymap(colorer_statut, subset=["Statut"]), use_container_width=True)

        if st.button("🗑️ Vider le journal"):
            st.session_state.journal_alertes = []
            st.rerun()