
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_service import get_answer

app = FastAPI(title="Teckinfo AI Voice Support")

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

@app.get("/")
def home():
    return {"message": "Teckinfo AI Voice Support API is running"}

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    try:
        answer = get_answer(request.question)
        return AnswerResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 