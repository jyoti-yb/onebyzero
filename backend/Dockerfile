FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MONSOONPP_BUNDLE=/opt/monsoonpp/bundle

WORKDIR /app
COPY requirements-runtime.lock ./
RUN pip install --no-cache-dir --no-deps -r requirements-runtime.lock && pip check
RUN mkdir -p /app/src/monsoonpp
COPY src/monsoonpp/__init__.py ./src/monsoonpp/
COPY src/monsoonpp/api ./src/monsoonpp/api
COPY src/monsoonpp/serving ./src/monsoonpp/serving

ARG BUNDLE_DIR=dist/demo-bundle
COPY ${BUNDLE_DIR}/ /opt/monsoonpp/bundle/
RUN python -c "from monsoonpp.serving import DemoBundle; DemoBundle('/opt/monsoonpp/bundle')"

USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()"
CMD ["uvicorn", "monsoonpp.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
