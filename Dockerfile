FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app

COPY requirements.txt requirements-live.txt ./

ARG INSTALL_LIVE_RETRIEVAL=false
RUN pip install --no-cache-dir -r requirements.txt && \
    if [ "$INSTALL_LIVE_RETRIEVAL" = "true" ]; then \
        pip install --no-cache-dir -r requirements-live.txt; \
    fi

COPY dashboard ./dashboard

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
