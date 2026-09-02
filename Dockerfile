FROM apache/airflow:2.9.3

RUN pip install --no-cache-dir \
    faker \
    psycopg2-binary \
    "numpy==1.26.4" \
    "pandas==2.1.4" \
    "scikit-learn==1.5.0" \
    joblib