FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

# headless OpenCV in the container; the pip package depends on the GUI build
# for the webcam window, so install the deps first and the package without them
RUN pip install --no-cache-dir opencv-python-headless numpy flask openvino pyyaml

COPY pyproject.toml README.md LICENSE /src/
COPY src /src/src
RUN pip install --no-cache-dir --no-deps /src && rm -rf /src

ENV HM_PASSES=/data/passes
VOLUME /data
EXPOSE 8080
HEALTHCHECK --interval=30s --start-period=60s \
    CMD curl -fs http://localhost:8080/healthz || exit 1
ENTRYPOINT ["haunted-mirror", "--headless"]
