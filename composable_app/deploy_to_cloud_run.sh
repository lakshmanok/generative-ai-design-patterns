#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- User-configurable names ---
export IMAGE_NAME="composable-app"
export SERVICE_NAME="composable-app-service"

# Attempt to get Project ID and Region from gcloud config.
export PROJECT_ID=$(gcloud config get-value project)
export REGION=$(gcloud config get-value compute/region)

# --- Validation ---
if [ -z "$PROJECT_ID" ]; then
    echo "ERROR: Google Cloud project ID is not set."
    echo "Please set it using 'gcloud config set project YOUR_PROJECT_ID'"
    exit 1
fi

if [ -z "$REGION" ]; then
    echo "ERROR: Google Cloud region is not set."
    echo "Please set it using 'gcloud config set compute/region YOUR_REGION'"
    exit 1
fi

echo "Using Project: $PROJECT_ID"
echo "Using Region: $REGION"

# --- Build with Cloud Build ---
echo "Building Docker image with Cloud Build..."
export IMAGE_TAG="gcr.io/${PROJECT_ID}/${IMAGE_NAME}"
gcloud builds submit --tag "${IMAGE_TAG}" .

# --- Deploy to Cloud Run ---
echo "Deploying container to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE_TAG}" \
  --platform "managed" \
  --region "${REGION}" \
  --allow-unauthenticated # Use --no-allow-unauthenticated for private services

echo "Deployment complete."
echo "Service URL will be displayed above."

