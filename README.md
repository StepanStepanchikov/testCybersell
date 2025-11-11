# ML Text Classifier API

Simple FastAPI service for text classification / sentiment analysis using Hugging Face or a mock local model.  

---

## 1. Setup

1. Clone the repo:

git clone https://github.com/StepanStepanchikov/testCybersell
cd testCybersell

2. Create and activate a virtual environment (optional but recommended):

python -m venv .venv

Windows \
.\.venv\Scripts\activate

Linux / Mac \
source .venv/bin/activate

3. Install dependencies:

pip install -r requirements.txt

## 2. Configure API key (if using Hugging Face)

1. Go to Hugging Face → Access Tokens -> Create new Token

2. Create a .env file in project root:

PROVIDER=hf # or 'mock' for local testing \
HF_API_URL=https://router.huggingface.co/hf-inference/models/distilbert/distilbert-base-uncased-finetuned-sst-2-english \
HF_API_TOKEN=<YOUR_HF_TOKEN> \
CACHE_TTL=3600 \
EXTERNAL_TIMEOUT=5

If you use mock, you can skip HF_API_TOKEN.

## 3. Run the API
Locally:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

With Docker:

docker build -t ml-text-classifier .
docker run -p 8000:8000 --env-file .env ml-text-classifier

## 4. Example Request
curl -X 'POST' \
  'http://127.0.0.1:8000/classify' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"text": "Good product"}'

Response:
```json 
{
  "labels": [
    {
      "label": "POSITIVE",
      "score": 0.9998588562011719
    },
    {
      "label": "NEGATIVE",
      "score": 0.0001411094854120165
    }
  ],
  "from_cache": false,
  "meta": {
    "provider": "hf"
  }
}
```

## 5. Health Check
curl http://localhost:8000/health

Response:
```json
{ 
  "status": "ok", 
  "provider": "hf" 
}
```
