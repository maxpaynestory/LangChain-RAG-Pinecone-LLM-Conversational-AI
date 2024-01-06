**In the name of ALLAH, who has guided me to do this work**

# LangChain RAG Pinecone Conversational AI news bot.
A conversational bot which impersonates Aljazeera English news Channel. Scraped a few pages from Aljazeera news website and fed it to LlaMA 2 using RAG. Pinecone is used as vector database.

![](/screenshots/who-are-you.PNG)
![](/screenshots/doing-rag-llm-predict.PNG)
![](/screenshots/llm-pinecone-working.PNG)
![](/screenshots/more-results-with-context.PNG)

### 1. Download LlaMA 2 model from Hugging Face

[LlaMA 2 7b Model GGUF file](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q8_0.gguf) 7 GB

### 2. Place downloaded file path inside llama_cpu.py

```python
MODEL_PATH = "D:\\path\\to\\folder\\llama-2-7b-chat.Q8_0.gguf"
```

### 3. Install Packages Python version is 3.9.0

```sh
$ pip install -r requirements.txt
```

### 4. Setup Pinecone
Signup to Pinecone and create a index with the name ```llama7bnewsbot```. The dimensions of index are ```4096``` and metric ```euclidean```.

After index is created place environment variables on your system as ```PINECONE_API_KEY``` and ```PINECONE_ENVIRONMENT```. These environment variables are used by pinecone client.

### 4. Fill Pinecone index
I have filled pinecone vector database using the news from Aljazeera channel english website. You can change the date for which you want to download the news for. To fill Pinecone database run.

```python llama_cpu.py```

That will download news from website and then fill them inside Pinecone vector database.
### 5. Run server
```sh
$ uvicorn api:app
```

### 6. Open frontend

Open browser and navigate to URL

http://127.0.0.1:8000/frontend/index.html

## CPU/RAM requirements

Atleast a good 6 core CPU.
12 GB of free RAM.