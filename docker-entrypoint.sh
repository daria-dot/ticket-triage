#!/bin/sh
# SageMaker runs a bring-your-own container as `docker run <image> serve`, and
# expects the model server on port 8080. Any other invocation runs the normal
# application API, so one image covers both.
set -e

if [ "$1" = "serve" ]; then
    exec uvicorn triage.api.sagemaker:app --host 0.0.0.0 --port 8080
fi

exec uvicorn triage.api.main:app --host 0.0.0.0 --port 8000
