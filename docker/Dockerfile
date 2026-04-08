FROM python:3.10

WORKDIR /app

# Heavy ML layers cached separately — rebuild won't re-download these
RUN pip install --timeout=300 torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --timeout=300 sentence-transformers faiss-cpu numpy

# App dependencies
COPY config/requirements.txt .
RUN pip install --timeout=300 -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]