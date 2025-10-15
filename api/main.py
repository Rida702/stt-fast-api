from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile
import requests
import base64
import os

app = FastAPI(
    title="Google Medical Speech-to-Text API",
    description="API for transcribing medical audio using Google Cloud Speech-to-Text (REST API version)",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI is live!"}

@app.get("/ping")
def ping():
    return {"status": "ok"}

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

@app.post("/transcribe", summary="Transcribe a medical audio file")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        if not GOOGLE_API_KEY:
            return JSONResponse({"status": "error", "message": "Missing Google API Key"})

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            content = await file.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name

        # Encode file to base64
        with open(temp_audio_path, "rb") as audio_file:
            audio_content = base64.b64encode(audio_file.read()).decode("utf-8")

        # Google Speech-to-Text endpoint
        url = f"https://speech.googleapis.com/v1p1beta1/speech:recognize?key={GOOGLE_API_KEY}"

        payload = {
            "config": {
                "encoding": "LINEAR16",
                "sampleRateHertz": 16000,
                "languageCode": "en-US",
                "useEnhanced": True,
                "model": "medical_conversation"
            },
            "audio": {"content": audio_content}
        }

        response = requests.post(url, json=payload)
        response_data = response.json()

        os.remove(temp_audio_path)

        if "results" in response_data:
            transcriptions = [
                result["alternatives"][0]["transcript"]
                for result in response_data["results"]
            ]
            full_text = " ".join(transcriptions)
            return JSONResponse({"status": "success", "transcription": full_text})

        return JSONResponse({
            "status": "error",
            "message": response_data.get("error", {}).get("message", "Unknown error")
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})
