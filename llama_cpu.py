import datetime
import os
from pyppeteer import launch
from pyppeteer.browser import Page
from pyppeteer_stealth import stealth
import asyncio
from bs4 import BeautifulSoup
import time
import glob
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores.pinecone import Pinecone
from langchain.schema import Document
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.retrievers import ContextualCompressionRetriever
import pinecone
from bs4 import BeautifulSoup
import html2text
import aiofiles as aiof
import bleach

MODEL_PATH = "D:\\llm-models\\TheBloke\\Llama-2-7b-Chat-GGUF\\llama-2-7b-chat.Q8_0.gguf"

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """You are an impersonator of Aljazeera English Channel.
                        Your goal is to help people answer their news related questions."""


def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    system_prompt = B_SYS + new_system_prompt + E_SYS
    prompt_template = B_INST + system_prompt + instruction + E_INST
    return prompt_template


def load_vector_db() -> Pinecone:
    pinecone.init()
    pinecone_index = pinecone.Index("llama7bnewsbot")
    embeddings = LlamaCppEmbeddings(model_path=MODEL_PATH, verbose=False)
    vectorstore = Pinecone(pinecone_index, embeddings, text_key="mytextkey")
    return vectorstore


def load_model():
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_threads=4,
        temperature=0.7,
        top_p=0.95,
        verbose=False,  # Verbose is required to pass to the callback manager
        n_ctx=2000,
    )
    return llm


def get_chain(llm):
    instruction = "{searched_context} \n\nUser: {user_input} \n\nAljazeera:"
    template = get_prompt(instruction)
    prompt = PromptTemplate(
        input_variables=["user_input", "searched_context"], template=template
    )
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False,
    )
    return llm_chain


def send_message(
    session,
    user_input,
    compression_retriever: ContextualCompressionRetriever,
    llm_chain: LLMChain,
    vector_store: Pinecone,
) -> str:
    docs = compression_retriever.get_relevant_documents(user_input)
    context = ""
    for docs in docs:
        context += docs.page_content
    bot_response = llm_chain.predict(user_input=user_input, searched_context=context)
    new_doc = Document(
        page_content="User: {user_input}\n\nAljazeera: {bot_response}".format(
            user_input=user_input, bot_response=bot_response
        ),
        metadata={"source": session},
    )
    vector_store.add_documents([new_doc])
    return bot_response


async def download_news_for_date(date: datetime.datetime, page: Page):
    if os.path.exists("downloads") is False:
        os.mkdir("downloads")
    download_folder = os.path.join("downloads", date.strftime("%Y-%m-%d"))
    if os.path.exists(download_folder) is False:
        os.mkdir(download_folder)
    else:
        num_text_files = len(glob.glob1(download_folder, "*.txt"))
        if num_text_files > 0:
            return download_folder
    aljazeera_search_url = "https://www.aljazeera.com/search/{date_string}".format(
        date_string=date.strftime("%d %B %Y")
    )
    options = {"waitUntil": "load", "timeout": 60000 * 5}  # 60000 * 10 is 5 minute
    await page.goto(aljazeera_search_url, options=options)
    content = await page.content()
    soup = BeautifulSoup(
        content,
        "html.parser",
    )
    links = soup.select("a.u-clickable-card__link")
    text_maker = html2text.HTML2Text()
    text_maker.ignore_links = True
    text_maker.bypass_tables = False
    text_maker.ignore_images = True
    for a in links:
        news_link = a["href"]
        await page.goto(news_link, options=options)
        content = await page.content()
        soup = BeautifulSoup(content, "html.parser")
        news_date = soup.select("div.article-dates")[0].prettify()
        news_title = soup.select("h1")[0].prettify()
        news_description = soup.select("div.wysiwyg")[0].prettify()
        news_text = text_maker.handle(news_date + news_title + news_description)
        news_text = bleach.clean(news_text)
        text_filename = "{folder}{sep}{pdf_filename}.txt".format(
            folder=download_folder, sep=os.sep, pdf_filename=time.time_ns()
        )
        out = await aiof.open(text_filename, "w", encoding="utf-8")
        await out.write(news_text)
        await out.flush()
    return download_folder


async def intercept(request):
    if any(request.resourceType == _ for _ in ("stylesheet", "image", "font")):
        await request.abort()
    else:
        await request.continue_()


async def init_pyppeteer() -> Page:
    browser = await launch({"autoClose": True})
    page = await browser.newPage()
    # await page.setRequestInterception(True)
    # page.on("request", lambda req: asyncio.ensure_future(intercept(req)))
    await stealth(page)
    return page


def fill_pinecone(download_folder):
    text_loader_kwargs = {"autodetect_encoding": True}
    loader = DirectoryLoader(
        download_folder,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs=text_loader_kwargs,
    )
    docs = loader.load()
    chunked_docs: list[Document] = []
    for doc in docs:
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=50,
            chunk_overlap=10,
            length_function=len,
            is_separator_regex=False,
        )
        new_docs = text_splitter.create_documents([doc.page_content], [doc.metadata])
        chunked_docs = chunked_docs + new_docs
    pinecone.init()
    pinecone_index = pinecone.Index("llama7bnewsbot")
    print(pinecone_index.describe_index_stats())
    embeddings = LlamaCppEmbeddings(model_path=MODEL_PATH, verbose=False)
    vectorstore = Pinecone(pinecone_index, embedding=embeddings, text_key="mytextkey")
    vectorstore.add_documents(documents=chunked_docs)


async def news_start_bot():
    page = await init_pyppeteer()
    download_folder = await download_news_for_date(datetime.datetime(2023, 11, 7), page)
    fill_pinecone(download_folder)


if __name__ == "__main__":
    asyncio.run(news_start_bot())
