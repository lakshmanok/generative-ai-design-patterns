#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- User-configurable names ---
export IMAGE_NAME="composable-app"
export SERVICE_NAME="composable-app-service"
export REPOSITORY="composable-app-repo" # Set your Artifact Registry repo name here

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

if [ -z "$GEMINI_API_KEY" ]; then
    echo "ERROR: GEMINI_API_KEY environment variable is not set."
    echo "Set it in your shell or pass it inline: GEMINI_API_KEY=your-key bash deploy_to_cloud_run.sh"
    echo "Your keys.env is not present in the Docker image (to avoid developer keys being used)"
    exit 1
fi

echo "Using Project: $PROJECT_ID"
echo "Using Region: $REGION"
echo "Using Artifact Registry Repository: $REPOSITORY"

# --- Build with Cloud Build ---
echo "Building Docker image with Cloud Build..."
export IMAGE_TAG="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}"
gcloud builds submit --tag "${IMAGE_TAG}" .

# --- Deploy to Cloud Run ---
# By default only people with the IAP-secured Web App User role (owners, editors have it already) in this project can invoke this service.
# See https://cloud.google.com/run/docs/securing/managing-access for other access options
echo "Deploying container to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE_TAG}" \
  --platform "managed" \
  --region "${REGION}" \
  --set-env-vars=GEMINI_API_KEY=${GEMINI_API_KEY}

echo "Deployment complete."
echo "Service URL will be displayed above."

