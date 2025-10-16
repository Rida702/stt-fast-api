from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from google.cloud import storage, speech
import tempfile
import os
import time

import json
from google.oauth2 import service_account

# Load service account from environment variable
service_account_info = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
credentials = service_account.Credentials.from_service_account_info(service_account_info)


app = FastAPI(
    title="Google Medical Speech-to-Text API",
    description="API for transcribing medical audio using Google Cloud Speech-to-Text (REST API version)",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI is live!"}


GCS_BUCKET_NAME = "scribe-ai-graham"


def upload_to_gcs(local_path, filename):
    print(f"[DEBUG] Uploading {filename} to GCS bucket {GCS_BUCKET_NAME}...")
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(filename)
    blob.upload_from_filename(local_path)
    gcs_uri = f"gs://{GCS_BUCKET_NAME}/{filename}"
    print(f"[DEBUG] File uploaded. GCS URI: {gcs_uri}")
    return gcs_uri


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        print("[DEBUG] Received file:", file.filename)
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            content = await file.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name
        print(f"[DEBUG] Saved temporary file at {temp_audio_path}")

        # Upload to GCS
        gcs_uri = upload_to_gcs(temp_audio_path, file.filename)
        os.remove(temp_audio_path)
        print("[DEBUG] Temporary file deleted after upload")

        # Initialize Speech client
        client = speech.SpeechClient(credentials=credentials)
        print("[DEBUG] Speech client initialized")

        audio = speech.RecognitionAudio(uri=gcs_uri)

        # Correct diarization config
        diarization_config = speech.SpeakerDiarizationConfig(
            enable_speaker_diarization=True,
            min_speaker_count=2,
            max_speaker_count=2
        )

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
            use_enhanced=True,
            model="medical_conversation",
            diarization_config=diarization_config
        )

        print("[DEBUG] Starting long-running recognition...")
        operation = client.long_running_recognize(config=config, audio=audio)

        # Wait for result
        print("[DEBUG] Waiting for operation to complete...")
        response = operation.result(timeout=600)
        print("[DEBUG] Transcription completed")

        full_text = " ".join([result.alternatives[0].transcript for result in response.results])
        print("[DEBUG] Combined transcript:", full_text[:100], "...")  # print first 100 chars

        return JSONResponse({"status": "success", "transcription": full_text})

    except Exception as e:
        print("[ERROR]", str(e))
        return JSONResponse({"status": "error", "message": str(e)})
