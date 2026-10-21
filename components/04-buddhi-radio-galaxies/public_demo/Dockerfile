FROM python:3.12-slim
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY --chown=user:user requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user:user app.py ./app.py
COPY --chown=user:user demo ./demo
USER user
ENV RADIOMORPH_PUBLIC_DEMO=1 GRADIO_ANALYTICS_ENABLED=False GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860 PYTHONDONTWRITEBYTECODE=1
EXPOSE 7860
CMD ["python", "-B", "app.py"]
