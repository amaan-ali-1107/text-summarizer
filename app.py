from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# Import pipeline
from src.textSummarizer.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI()

# Setup Jinja2 templates
templates = Jinja2Templates(directory="template")

# Initialize once
pipeline = PredictionPipeline()

# Pre-load templates on startup
@app.on_event("startup")
async def startup_event():
    # Pre-compile the summary template
    templates.get_template("summary.html")
    print("Templates pre-loaded!")

# Pydantic model for JSON requests
class TextRequest(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/summarize", response_class=HTMLResponse)
async def summarize(request: Request, text: str = Form(...)):
    try:
        summary = pipeline.predict(text)
        return templates.TemplateResponse("summary.html", {
            "request": request, 
            "summary": summary,
            "original_text": text
        })
    except Exception as e:
        return templates.TemplateResponse("summary.html", {
            "request": request, 
            "summary": f"Error: {e}",
            "original_text": text
        })

@app.post("/predict")
async def predict_route(req: TextRequest):
    try:
        summary = pipeline.predict(req.text)
        return {"summary": summary}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)