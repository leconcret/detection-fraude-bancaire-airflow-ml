-- ============================================================
-- SCHEMA.SQL
-- Projet : Detection de Fraude Bancaire
-- Base de donnees : banque_fraude (PostgreSQL)
-- ============================================================


-- ============================================================
-- 1. CREATION DES TABLES
-- ============================================================

-- Table des clients
CREATE TABLE clients (
    client_id SERIAL PRIMARY KEY,
    nom VARCHAR(100) NOT NULL,
    date_ouverture_compte DATE NOT NULL,
    segment VARCHAR(20) NOT NULL CHECK (segment IN ('Particulier', 'Professionnel'))
);

-- Table des comptes
CREATE TABLE comptes (
    compte_id SERIAL PRIMARY KEY,
    client_id INTEGER NOT NULL REFERENCES clients(client_id),
    type_compte VARCHAR(20) NOT NULL CHECK (type_compte IN ('Courant', 'Epargne')),
    solde NUMERIC(12, 2) NOT NULL DEFAULT 0
);

-- Table des transactions
CREATE TABLE transactions (
    transaction_id SERIAL PRIMARY KEY,
    compte_id INTEGER NOT NULL REFERENCES comptes(compte_id),
    montant NUMERIC(12, 2) NOT NULL,
    type_transaction VARCHAR(20) NOT NULL CHECK (type_transaction IN ('Retrait', 'Depot', 'Virement', 'Paiement')),
    date_transaction TIMESTAMP NOT NULL,
    pays VARCHAR(50) NOT NULL,
    est_fraude BOOLEAN NOT NULL DEFAULT FALSE
);


-- ============================================================
-- 2. VUE DE FEATURE ENGINEERING (fonctions de fenetre)
-- ============================================================
-- Cette vue calcule, pour chaque transaction, des variables
-- comportementales utilisees comme features du modele ML :
--   - montant_moyen_historique : moyenne des transactions PRECEDENTES
--     du meme compte (exclut la transaction actuelle -> evite le data leakage)
--   - nb_transactions_24h : nombre de transactions du compte dans une
--     fenetre glissante des dernieres 24h
--   - premiere_fois_ce_pays : indicateur binaire (1/0) si c'est la
--     premiere transaction du compte dans ce pays
--   - heure_transaction : heure extraite du timestamp (0-23)
--   - segment : recupere par jointure transactions -> comptes -> clients

CREATE OR REPLACE VIEW vue_features_transactions AS
SELECT
  t.transaction_id,
  t.compte_id,
  t.date_transaction,
  t.montant,
  t.type_transaction,
  t.pays,
  ROUND(AVG(t.montant) OVER (
    PARTITION BY t.compte_id
    ORDER BY t.date_transaction
    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
  ), 2) AS montant_moyen_historique,
  COUNT(*) OVER (
    PARTITION BY t.compte_id
    ORDER BY t.date_transaction
    RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW
  ) AS nb_transactions_24h,
  CASE WHEN ROW_NUMBER() OVER (
    PARTITION BY t.compte_id, t.pays
    ORDER BY t.date_transaction
  ) = 1 THEN 1 ELSE 0 END AS premiere_fois_ce_pays,
  EXTRACT(HOUR FROM t.date_transaction) AS heure_transaction,
  c.segment,
  t.est_fraude
FROM transactions t
JOIN comptes cp ON t.compte_id = cp.compte_id
JOIN clients c ON cp.client_id = c.client_id;


-- ============================================================
-- 3. REQUETES DE VERIFICATION UTILES
-- ============================================================

-- Comptage global
SELECT
  (SELECT COUNT(*) FROM clients) AS nb_clients,
  (SELECT COUNT(*) FROM comptes) AS nb_comptes,
  (SELECT COUNT(*) FROM transactions) AS nb_transactions,
  (SELECT COUNT(*) FROM transactions WHERE est_fraude) AS nb_fraudes;

-- Verification de la vue
SELECT COUNT(*) FROM vue_features_transactions;

-- Exemple : detail des transactions frauduleuses
SELECT * FROM transactions WHERE est_fraude = true LIMIT 10;