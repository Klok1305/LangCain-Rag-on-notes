from __future__ import annotations
from pathlib import Path
from typing import Tuple, List, Optional

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from ddgs import DDGS

# ==== дефолты (должны совпадать с ingest.py) ====
LLM_MODEL = "gemma3n:e2b"
EMB_MODEL = "nomic-embed-text"
COLLECTION = "local_docs"
RETRIEVER_K = 5
RELEVANCE_THRESHOLD = 0.55       # по желанию включается
WEB_MAX_RESULTS = 5
WEB_TIMEOUT = 10

# ==== LLM и векторка ====
def build_llm() -> ChatOllama:
    return ChatOllama(model=LLM_MODEL, temperature=0.2, num_ctx=4096)

def build_vectorstore(persist_dir: str | Path,
                      collection_name: Optional[str] = None,
                      emb_model: Optional[str] = None) -> Chroma:
    return Chroma(
        embedding_function=OllamaEmbeddings(model=emb_model or EMB_MODEL),
        persist_directory=str(persist_dir),
        collection_name=collection_name or COLLECTION,
    )

def build_retriever(persist_dir: str | Path,
                    k: Optional[int] = None,
                    threshold: Optional[float] = None):
    """
    Если threshold is None — обычный similarity top-k (без порога).
    Если threshold задан — similarity_score_threshold.
    """
    vs = build_vectorstore(persist_dir)
    k = int(k or RETRIEVER_K)
    if threshold is None:
        return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
    else:
        return vs.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": float(threshold), "k": k}
        )

def format_docs(docs) -> str:
    return "\n\n".join(getattr(d, "page_content", "") for d in (docs or []))

def prompt_chain(llm: ChatOllama):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Ты локальный ассистент. Опирайся на предоставленный контекст. "
         "Если локального ответа нет, используй веб-контекст. "
         "Отвечай кратко и по делу."),
        ("human", "Вопрос: {question}\n\nКонтекст:\n{context}")
    ])
    return prompt | llm | StrOutputParser()

# ==== веб-fallback (устойчивый к разным сигнатурам ddgs) ====
def web_search(query: str, max_results: int = WEB_MAX_RESULTS) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    try:
        with DDGS(timeout=WEB_TIMEOUT) as ddg:
            gen = None
            try:
                gen = ddg.text(query, max_results=max_results)  # новая сигнатура
            except TypeError:
                gen = ddg.text(keywords=query, max_results=max_results)  # старая сигнатура
            for r in gen:
                out.append((r.get("title", ""), r.get("url", ""), r.get("body", "")))
    except Exception as e:
        print("[web_search] error:", e)
    return out

def build_web_context(query: str, k: int = WEB_MAX_RESULTS) -> Tuple[str, List[str]]:
    hits = web_search(query, max_results=k)
    if not hits:
        return "", []
    blocks, urls = [], []
    for i, (title, url, body) in enumerate(hits, 1):
        blocks.append(f"[{i}] {title}\n{body}\nURL: {url}")
        urls.append(url)
    return "\n\n".join(blocks), urls

# ==== высокоуровневый вызов ====
def answer(question: str,
           persist_dir: str | Path,
           k: Optional[int] = None,
           use_threshold: bool = False,
           threshold: Optional[float] = None,
           web_k: Optional[int] = None) -> tuple[str, str, list[dict]]:
    """
    Возвращает (text, src_kind, cites)
      src_kind: 'local' | 'web' | 'none'
      cites: [{'type': 'local'|'web', 'label': str}]
    """
    llm = build_llm()
    chain = prompt_chain(llm)

    retr = build_retriever(
        persist_dir=persist_dir,
        k=k,
        threshold=(threshold if use_threshold else None)
    )
    docs = retr.invoke(question)

    if docs:
        ctx = format_docs(docs)
        cites = []
        for d in docs:
            meta = getattr(d, "metadata", {}) or {}
            src = meta.get("source", "unknown")
            page = meta.get("page", None)
            label = f"{Path(src).name}{f': p.{page}' if page is not None else ''}"
            cites.append({"type": "local", "label": label})
        src_kind = "local"
    else:
        web_ctx, urls = build_web_context(question, k=int(web_k or WEB_MAX_RESULTS))
        ctx = web_ctx if web_ctx else "Контекст отсутствует."
        cites = [{"type": "web", "label": u} for u in urls] if urls else []
        src_kind = "web" if urls else "none"

    text = chain.invoke({"question": question, "context": ctx})
    return text, src_kind, cites

if __name__ == "__main__":
    import sys
    pdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("chroma")
    q = input("> ").strip()
    t, s, c = answer(q, persist_dir=pdir)
    print(f"[{s}] {t}")
    if c:
        print("Источники:")
        for x in c:
            print("-", x["label"])
