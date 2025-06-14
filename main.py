from fastapi import FastAPI
from pydantic import BaseModel
from build_answer import get_answer

app = FastAPI()

class Question(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "Virtual TA is running. Use the /answer endpoint with a POST request."}

@app.post("/answer")
def answer(data: Question):
    try:
        response = get_answer(data.question)
        return {"question": data.question, "answer": response}
    except Exception as e:
        return {"error": str(e)}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


