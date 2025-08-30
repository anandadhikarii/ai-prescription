# main.py
import os, json, re
from typing import List, Dict, Any, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import uvicorn

load_dotenv()  # loads .env if present

app = FastAPI()

# --- CORS (leave wide-open while developing) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load your local drug database ---
def load_drug_data() -> List[Dict[str, Any]]:
    try:
        with open("drugs.json", "r", encoding="utf-8") as f:
            return json.load(f).get("drugs", [])
    except FileNotFoundError:
        return []

drug_database = load_drug_data()

# --- Hugging Face Granite config ---
HF_TOKEN = os.getenv("HF_TOKEN")
GRANITE_MODEL = os.getenv("GRANITE_MODEL", "ibm-granite/granite-3.3-2b-instruct")
GRANITE_ENDPOINT_URL = os.getenv("GRANITE_ENDPOINT_URL")  # If set, use endpoint (recommended)

# Build a client that works with either a deployed endpoint (base_url)
# or, if not provided, tries serverless with the model name.
if GRANITE_ENDPOINT_URL:
    # OpenAI-compatible base_url; don't pass model here
    hf_client = InferenceClient(base_url=GRANITE_ENDPOINT_URL, api_key=HF_TOKEN)
else:
    # Route through HF with the model name (may fail if model isn't serverless-deployed)
    hf_client = InferenceClient(token=HF_TOKEN)

class PrescriptionRequest(BaseModel):
    prescription_text: str
    patient_age: int

def _best_json_chunk(text: str) -> Optional[str]:
    """Safely extract a JSON object from an LLM response."""
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    return match.group(0)

def extract_with_granite(prescription_text: str) -> List[Dict[str, Optional[str]]]:
    """
    Ask Granite to return a strict JSON structure:
    {
      "drugs":[{"name":"...", "dose":"...", "unit":"...", "frequency":"..."}]
    }
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You extract medication data from free text. "
                "Return ONLY strict JSON with keys: drugs -> list of objects "
                "[name, dose, unit, frequency]. Use null if unknown. No extra text."
            ),
        },
        {
            "role": "user",
            "content": f"Text: ```{prescription_text}```",
        },
    ]

    # Call using OpenAI-compatible chat completions
    if GRANITE_ENDPOINT_URL:
        out = hf_client.chat.completions.create(
            messages=messages,
            max_tokens=400,
            temperature=0.2,
        )
    else:
        # When not using endpoint, pass the model name explicitly
        out = hf_client.chat.completions.create(
            model=GRANITE_MODEL,
            messages=messages,
            max_tokens=400,
            temperature=0.2,
        )

    content = out.choices[0].message.content
    json_chunk = _best_json_chunk(content)
    if not json_chunk:
        return []

    try:
        data = json.loads(json_chunk)
        drugs = data.get("drugs", [])
        # normalize fields to strings-or-None
        norm = []
        for d in drugs:
            norm.append({
                "name": (d.get("name") or "").strip(),
                "dose": (d.get("dose") or None),
                "unit": (d.get("unit") or None),
                "frequency": (d.get("frequency") or None),
            })
        return norm
    except Exception:
        return []

def fuzzy_match_local(name: str) -> Optional[Dict[str, Any]]:
    """Case-insensitive contains/alias match against your local drug DB."""
    n = name.lower().strip()
    for drug in drug_database:
        if drug.get("name", "").lower() == n:
            return drug
        # Optional: match aliases field if present
        for alias in drug.get("aliases", []):
            if alias.lower() == n:
                return drug
        # Last resort: substring contains
        if n and n in drug.get("name", "").lower():
            return drug
    return None

@app.post("/analyze/")
def analyze_prescription(request: PrescriptionRequest):
    """
    Uses IBM Granite (via Hugging Face) to parse meds from free text.
    Then joins with your local `drugs.json` to return the same structure
    your UI expects: {"drugs": [...]}.
    On failure, falls back to simple keyword matching.
    """
    extracted: List[Dict[str, Optional[str]]] = []
    try:
        if HF_TOKEN:
            extracted = extract_with_granite(request.prescription_text)
    except Exception:
        extracted = []  # Granite call failed; continue with fallback

    found_drugs: List[Dict[str, Any]] = []

    if extracted:
        # Join Granite results to your local DB so the UI fields (interactions, alternatives...) still work
        for item in extracted:
            local = fuzzy_match_local(item["name"])
            if local:
                # Optionally annotate dosage/frequency into the returned object
                enriched = dict(local)
                enriched["dosage_recommendation"] = item.get("dose") and item.get("unit") \
                    and f'{item["dose"]} {item["unit"]}' or (local.get("dosage_recommendation") or "Not specified")
                # You can also stash frequency in a new key if you want
                enriched["parsed_frequency"] = item.get("frequency")
                found_drugs.append(enriched)
    else:
        # Fallback to your original simple approach
        for drug in drug_database:
            if drug["name"].lower() in request.prescription_text.lower():
                found_drugs.append(drug)

    return {"drugs": found_drugs}

if __name__ == "__main__":
    # Running with reload during dev:
    uvicorn.run(app, host="0.0.0.0", port=8000)
