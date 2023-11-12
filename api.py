from typing import Annotated
from fastapi import APIRouter, FastAPI, Form
from fastapi.staticfiles import StaticFiles
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from llama_cpu import get_chain, load_model, load_vector_db, send_message


app = FastAPI()
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
router = APIRouter()

llm = load_model()
app.llm_chain = get_chain(llm)
app.vector_store = load_vector_db()
retriever = app.vector_store.as_retriever(search_type="mmr")
compressor = LLMChainExtractor.from_llm(llm)
app.compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)


@router.post("/message")
async def post_message(
    user_input: Annotated[str, Form()], session: Annotated[str, Form()]
):
    bot_repy = send_message(
        session, user_input, app.compression_retriever, app.llm_chain, app.vector_store
    )
    return {"botResponse": bot_repy}


app.include_router(router, prefix="/api")
