FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate synthetic data and train all models at build time
RUN python src/utils/data_generator.py && \
    python src/credit_scoring.py && \
    python src/fraud_detection.py && \
    python src/expense_categorizer.py

EXPOSE 5000

CMD ["python", "src/app.py"]
