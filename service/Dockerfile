FROM tensorflow/serving
WORKDIR /
COPY models /models
COPY models.config /models/models.config
EXPOSE 8501
CMD ["--model_config_file=/models/models.config"]
