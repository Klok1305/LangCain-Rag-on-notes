from __future__ import annotations

from typing import Tuple
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from ddgs import DDGS

# ===== настройки по умолчанию =====
LLM_MODEL = "gemma3n:e2b"
EMB_MODEL = "nomic-embed-text"
CHROMA_DIR = "chroma"
COLLECTION = "local_docs"
RETRIEVER_K = 5
RELEVANCE_THRESHOLD = 0.65  
WEB_MAX_RESULTS = 5
WEB_TIMEOUT = 10  

# ===== общие компоненты =====

def build_llm() -> ChatOllama:
    return ChatOllama(model=LLM_MODEL, temperature=0.2, num_ctx=4096)

def build_retriever():
    embeddings = OllamaEmbeddings(model=EMB_MODEL)
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION,
    )
    return vectordb.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": RELEVANCE_THRESHOLD, "k": RETRIEVER_K}
    )

def format_docs(docs) -> str:
    return "\n\n".join(getattr(d, "page_content", "") for d in docs)

def prompt_chain(llm: ChatOllama):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Ты ассистент и отвечаешь, опираясь на предоставленный контекст. "
         "Отвечай кратко и по делу. Если в локальных документах нет ответа — используй веб-контекст. "
         ),
        ("human", "Вопрос: {question}\n\nКонтекст:\n{context}")
    ])
    return prompt | llm | StrOutputParser()

# ===== веб-поиск (fallback) =====

def web_search(query: str, max_results: int = WEB_MAX_RESULTS) -> list[tuple[str, str, str]]:
    """DuckDuckGo через ddgs с поддержкой обеих сигнатур .text(). Возвращает (title, url, body)."""
    results: list[tuple[str, str, str]] = []
    try:
        with DDGS(timeout=WEB_TIMEOUT) as ddg:
            gen = ddg.text(query, max_results=max_results)          
            for r in gen:
                results.append((
                    r.get("title", ""),
                    r.get("url", ""),
                    r.get("body", "")
                ))
    except Exception as e:
        print("[web_search] error:", e)
        return []
    return results

def build_web_context(query: str, k: int = WEB_MAX_RESULTS) -> Tuple[str, list[str]]:
    """Собирает читаемый блок контекста и список URL."""
    hits = web_search(query, max_results=k)
    if not hits:
        return "", []
    blocks, urls = [], []
    for i, (title, url, body) in enumerate(hits, 1):
        blocks.append(f"[{i}] {title}\n{body}\nURL: {url}")
        urls.append(url)
    return "\n\n".join(blocks), urls

# ===== оркестрация =====

def answer(question: str) -> str:
    llm = build_llm()
    retriever = build_retriever()
    chain = prompt_chain(llm)

    docs = retriever.invoke(question)
    if docs:
        context = format_docs(docs)
        source = "local"
    else:
        web_ctx, _ = build_web_context(question, k=WEB_MAX_RESULTS)
        context = web_ctx if web_ctx else "Контекст отсутствует."
        source = "web" if web_ctx else "none"

    print("Источник ответа:", source, "| Документов:", len(docs))
    return chain.invoke({"question": question, "context": context})


if __name__ == "__main__":
    print("Задайте вопрос:")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        try:
            print(answer(q))
        except Exception as e:
            print("Ошибка:", e)
