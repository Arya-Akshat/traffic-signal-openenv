FROM ghcr.io/astral-sh/uv:python3.11-bookworm

WORKDIR /app
RUN echo "BUILD $(date)"
RUN echo "FORCE BUILD $(date)"
COPY . /app

RUN python -m pip install -r requirements.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
