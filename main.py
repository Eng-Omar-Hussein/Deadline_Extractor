import re
import dateparser
from datetime import datetime
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import AutoTokenizer, pipeline

# ---------------------------------------------------------
# 1. The Deterministic Math Engine
# ---------------------------------------------------------
class AnchorParser:
    def __init__(self):
        self.fuzzy_mappings = {
            r'\b(?:EOD|COB|EOB|end of the day|end of day)\b': 'at 17:00',
            r'\b(?:EOW|end of the week)\b': 'Friday at 17:00',
            r'\b(?:ASAP|immediately|right away)\b': 'in 2 hours',
            r'\b(?:a few days|few days|a couple of days)\b': 'in 3 days',
            r'\b(?:the coming day|next day)\b': 'in 1 day'
        }
        self.ambiguous_days = {
            r'\b(?:tomorrow)\b': 'tomorrow at 17:00',
            r'\b(?:today)\b': 'today at 17:00',
            r'\b(?:tonight)\b': 'today at 20:00'
        }

    def _has_explicit_time(self, text):
        time_pattern = r'\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)|\d{1,2}:\d{2}'
        return bool(re.search(time_pattern, text, re.IGNORECASE))

    def _apply_logical_assumptions(self, candidate_string):
        cleaned = candidate_string
        for pattern, replacement in self.fuzzy_mappings.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        if not self._has_explicit_time(cleaned):
            for pattern, replacement in self.ambiguous_days.items():
                cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+(?:at|@)\s+', ' ', cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def resolve_deadline(self, candidate_string, sent_date_obj):
        math_ready_string = self._apply_logical_assumptions(candidate_string)
        settings = {
            'RELATIVE_BASE': sent_date_obj,
            'PREFER_DATES_FROM': 'future',
            'TIMEZONE': 'UTC',
            'RETURN_AS_TIMEZONE_AWARE': True,
            'STRICT_PARSING': False
        }
        final_datetime = dateparser.parse(math_ready_string, settings=settings)
        return final_datetime.isoformat() if final_datetime else None

# ---------------------------------------------------------
# 2. API Setup & Data Models
# ---------------------------------------------------------
app = FastAPI(title="Deadline Extraction API (Docker/ONNX Edition)")

class EmailRequest(BaseModel):
    email_text: str
    sent_date: datetime

class DeadlineResult(BaseModel):
    raw_text: str
    parsed_timestamp: str
    confidence_score: float

class EmailResponse(BaseModel):
    status: str
    extracted_deadlines: List[DeadlineResult]

# ---------------------------------------------------------
# 3. Model Loading (ONNX Runtime)
# ---------------------------------------------------------
MODEL_DIR = "./model"

print("Booting up ONNX INT8 Runtime...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    
    # Force tokenizer to drop token_type_ids for DistilBERT ONNX compatibility
    if "token_type_ids" in tokenizer.model_input_names:
        tokenizer.model_input_names.remove("token_type_ids")

    model = ORTModelForTokenClassification.from_pretrained(
        MODEL_DIR, 
        file_name="model.onnx"
    )
    
    ner_pipeline = pipeline(
        "ner", 
        model=model, 
        tokenizer=tokenizer, 
        aggregation_strategy="simple"
    )
    print("ONNX Engine loaded successfully!")
except Exception as e:
    print(f"Error loading ONNX model: {e}")

parser = AnchorParser()

# ---------------------------------------------------------
# 4. Core Endpoint
# ---------------------------------------------------------
@app.post("/extract", response_model=EmailResponse)
async def extract_deadlines(request: EmailRequest):
    try:
        extracted_entities = ner_pipeline(request.email_text)
        final_deadlines_processed = []
        
        for entity in extracted_entities:
            if entity.get("entity_group") == "DEADLINE":
                raw_extracted_text = entity.get("word").strip()
                confidence = float(entity.get("score"))
                
                timestamp = parser.resolve_deadline(raw_extracted_text, request.sent_date)
                
                if timestamp:
                    final_deadlines_processed.append(
                        DeadlineResult(
                            raw_text=raw_extracted_text,
                            parsed_timestamp=timestamp,
                            confidence_score=round(confidence * 100, 2)
                        )
                    )
                    
        return EmailResponse(status="success", extracted_deadlines=final_deadlines_processed)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))