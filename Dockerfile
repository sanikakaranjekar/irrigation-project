FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the model at build time so the image is self-contained.
# (For a real product you'd train separately and just COPY model.joblib in,
# but for a college project this keeps everything reproducible in one step.)
RUN python train_model.py

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
