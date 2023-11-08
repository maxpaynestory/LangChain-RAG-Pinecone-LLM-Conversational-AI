from typing import Annotated
from fastapi import APIRouter, FastAPI, Form
from fastapi.staticfiles import StaticFiles

from llama_cpu import load_model, load_vector_db, send_message


app = FastAPI()
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
router = APIRouter()

app.llm_chain = load_model()
app.vector_store = load_vector_db()


@router.post("/message")
async def post_message(
    user_input: Annotated[str, Form()], session: Annotated[str, Form()]
):
    bot_repy = send_message(session, user_input, app.llm_chain, app.vector_store)
    return {"botResponse": bot_repy}


app.include_router(router, prefix="/api")
