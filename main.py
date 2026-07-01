import re
import dateparser
from datetime import datetime
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import DistilBertTokenizerFast, pipeline

# ---------------------------------------------------------
# 1. Generator & Parser Configuration
# ---------------------------------------------------------
class CandidateGenerator:
    def __init__(self):
        patterns = [
            r'\b\d{1,4}[/\-\.]\d{1,2}[/\-\.]\d{1,4}\b',
            r'\b[a-z]{3,9}\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{2,4})?\b',
            r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?[a-z]{3,9}\.?(?:,?\s+\d{2,4})?\b',
            r'\b[a-z]{2,6}days?\b',
            r'\b\d{1,2}(?::\d{2}){0,2}\s*(?:[aApP]\.?[mM]\.?|hrs|hours)\b',
            r'\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*[A-Z]{3,4})?\b',
            r'\b(?:in|after|within)\s+(?:a|an|\d+)\s+(?:sec|min|hour|day|week|month|year)s?\b',
            r'\b(?:next|this|last|coming|upcoming|end\sof)\s+[a-z]+\b',
            r'\b(?:to(?:day|night|morrow)|yesterday|now|asap|eod|cob|eob|eow)\b'
        ]
        self.master_regex = re.compile('|'.join(patterns), re.IGNORECASE)
        self.connector_regex = re.compile(r'^\s*(?:at|@|on|by|,|and|-|to|\s)\s*$', re.IGNORECASE)

    def _merge_intervals(self, matches, text):
        if not matches: return []
        matches.sort(key=lambda x: x[0])
        merged = []
        curr_start, curr_end = matches[0][0], matches[0][1]
        for next_start, next_end in matches[1:]:
            between_text = text[curr_end:next_start]
            if next_start <= curr_end or self.connector_regex.match(between_text):
                curr_end = max(curr_end, next_end)
            else:
                merged.append((curr_start, curr_end, text[curr_start:curr_end].strip()))
                curr_start, curr_end = next_start, next_end
        merged.append((curr_start, curr_end, text[curr_start:curr_end].strip()))
        return merged

    def mask_email(self, email_text: str):
        raw_matches = [(m.start(), m.end()) for m in self.master_regex.finditer(email_text)]
        merged_candidates = self._merge_intervals(raw_matches, email_text)
        
        masked_text = ""
        mappings = [] 
        last_idx = 0
        
        for start, end, raw_text in merged_candidates:
            masked_text += email_text[last_idx:start]
            mask_start = len(masked_text)
            masked_text += "[TIME_REF]"
            mask_end = len(masked_text)
            
            mappings.append({
                "mask_start": mask_start,
                "mask_end": mask_end,
                "raw_text": raw_text
            })
            last_idx = end
            
        masked_text += email_text[last_idx:]
        return masked_text, mappings

class AnchorParser:
    def __init__(self):
        self.fuzzy_mappings = {
            r'\b(?:EOD|COB|EOB|end of the day|end of day)\b': 'at 17:00',
            r'\b(?:EOW|end of the week)\b': 'Friday at 17:00',
            r'\b(?:ASAP|immediately|right away)\b': 'in 2 hours',
            r'\b(?:the coming day|next day)\b': 'in 1 day'
        }
        self.ambiguous_days = {
            r'\b(?:tomorrow)\b': 'tomorrow at 17:00',
            r'\b(?:today)\b': 'today at 17:00',
            r'\b(?:tonight)\b': 'today at 20:00'
        }

    def resolve_deadline(self, candidate_string, sent_date_obj):
        cleaned = candidate_string
        for pattern, replacement in self.fuzzy_mappings.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        for pattern, replacement in self.ambiguous_days.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        settings = {'RELATIVE_BASE': sent_date_obj, 'PREFER_DATES_FROM': 'future', 'TIMEZONE': 'UTC', 'RETURN_AS_TIMEZONE_AWARE': True}
        final_datetime = dateparser.parse(cleaned, settings=settings)
        return final_datetime.isoformat() if final_datetime else None

# ---------------------------------------------------------
# 2. FastAPI Setup & ONNX Initialization
# ---------------------------------------------------------
app = FastAPI(title="Production ONNX Deadline API")
generator = CandidateGenerator()
parser = AnchorParser()

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

MODEL_DIR = "./model"
print("Booting up ONNX INT8 Engine...")
try:
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    if "token_type_ids" in tokenizer.model_input_names:
        tokenizer.model_input_names.remove("token_type_ids")

    model = ORTModelForTokenClassification.from_pretrained(MODEL_DIR, file_name="model.onnx")
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    print("ONNX Engine active!")
except Exception as e:
    print(f"Error loading ONNX model: {e}")

# ---------------------------------------------------------
# 3. Execution Endpoint
# ---------------------------------------------------------
@app.post("/extract", response_model=EmailResponse)
async def extract_deadlines(request: EmailRequest):
    try:
        masked_text, mappings = generator.mask_email(request.email_text)
        extracted_entities = ner_pipeline(masked_text)
        final_deadlines = []
        
        for entity in extracted_entities:
            if entity.get("entity_group") == "DEADLINE":
                ent_start, ent_end = entity["start"], entity["end"]
                
                for mapping in mappings:
                    if max(mapping["mask_start"], ent_start) < min(mapping["mask_end"], ent_end):
                        raw_text = mapping["raw_text"]
                        confidence = float(entity.get("score"))
                        
                        timestamp = parser.resolve_deadline(raw_text, request.sent_date)
                        if timestamp:
                            final_deadlines.append(
                                DeadlineResult(raw_text=raw_text, parsed_timestamp=timestamp, confidence_score=round(confidence * 100, 2))
                            )
                        break 
                        
        return EmailResponse(status="success", extracted_deadlines=final_deadlines)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
