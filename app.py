# app.py — положи в КОРЕНЬ проекта
import sys, importlib
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import rag as rag_mod
try:
    import ingest as ingest_mod
except Exception:
    ingest_mod = None

DATA = ROOT / "data"
PDF = DATA / "pdf"
TXT = DATA / "txt"
CHROMA_DIR = ROOT / "chroma"

st.set_page_config(page_title="Local RAG", page_icon="💬", layout="wide")
st.title("Local RAG")
st.caption("Документы: data/pdf, data/txt. Эмбеддинги: chroma/. Полный диалог и устойчивый веб-fallback.")

# ===== session state =====
if "history" not in st.session_state:
    st.session_state.history = []  # [{role:'user'|'assistant', content:str, sources:list}]

# ===== sidebar: настройки + файлы + переиндексация =====
with st.sidebar:
    st.subheader("Настройки запроса")
    k_default = int(getattr(rag_mod, "RETRIEVER_K", 5))
    thr_default = float(getattr(rag_mod, "RELEVANCE_THRESHOLD", 0.55))
    web_default = int(getattr(rag_mod, "WEB_MAX_RESULTS", 5))

    k = st.slider("k (число чанков)", 1, 12, k_default)
    use_thr = st.checkbox("Использовать порог релевантности", value=False)
    threshold = st.slider("Порог (если включён)", 0.0, 1.0, thr_default, step=0.05)
    web_k = st.slider("Кол-во веб-результатов", 1, 10, web_default)

    st.markdown("---")
    st.subheader("Добавить файлы")
    up = st.file_uploader("Перетащи .pdf / .txt / .md", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if up:
        saved = []
        for f in up:
            name = Path(f.name).name
            if name.lower().endswith(".pdf"):
                PDF.mkdir(parents=True, exist_ok=True)
                (PDF / name).write_bytes(f.read())
                saved.append(f"data/pdf/{name}")
            else:
                TXT.mkdir(parents=True, exist_ok=True)
                (TXT / name).write_bytes(f.read())
                saved.append(f"data/txt/{name}")
        st.success("Сохранено:\n" + "\n".join(saved))

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔁 Переиндексировать"):
            if ingest_mod is None:
                st.error("src/ingest.py не найден")
            else:
                try:
                    importlib.reload(ingest_mod)
                    ingest_mod.main()  # читает data/*, пишет в chroma/
                    st.success("Индекс обновлён в chroma/")
                except Exception as e:
                    st.error(f"Ошибка индексации: {e}")
    with col_b:
        if st.button("♻️ Перезагрузить rag"):
            try:
                importlib.reload(rag_mod)
                st.success("rag.py перезагружен")
            except Exception as e:
                st.error(f"Ошибка: {e}")

    st.markdown("---")
    if st.button("🗑 Очистить диалог"):
        st.session_state.history = []
        st.success("История очищена")

# ===== статус наличия данных/индекса =====
if not DATA.exists():
    st.warning("Папка data/ не найдена. Создай data/pdf и data/txt или загрузите файлы слева.")
if not CHROMA_DIR.exists() or not (CHROMA_DIR / "chroma.sqlite3").exists():
    st.warning("Индекс Chroma не найден. Нажми 'Переиндексировать' слева после добавления файлов.")

# ===== вывод всей истории =====
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Источники", expanded=False):
                for s in msg["sources"]:
                    st.write(("🗂 " if s["type"] == "local" else "🌐 ") + s["label"])

# ===== ввод и ответ =====
q = st.chat_input("Вопрос")
if q:
    # 1) добавляем юзера в историю
    st.session_state.history.append({"role": "user", "content": q})

    # 2) показываем сообщение юзера
    with st.chat_message("user"):
        st.markdown(q)

    # 3) считаем ответ
    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            try:
                importlib.reload(rag_mod)  # подхватываем свежие правки rag.py
                text, src_kind, cites = rag_mod.answer(
                    question=q,
                    persist_dir=CHROMA_DIR,
                    k=k,
                    use_threshold=use_thr,
                    threshold=threshold,
                    web_k=web_k
                )
            except Exception as e:
                text, src_kind, cites = f"Ошибка: {e}", "none", []

        # 4) показываем ассистента и источники
        st.markdown(text)
        if cites:
            label = {"local": "Локальные документы", "web": "Веб", "none": "Нет контекста"}.get(src_kind, "Источники")
            with st.expander(f"Источники • {label}", expanded=False):
                for s in cites:
                    st.write(("🗂 " if s["type"] == "local" else "🌐 ") + s["label"])

    # 5) добавляем ассистента в историю (важно: ПОСЛЕ отображения, но в ту же сессию)
    st.session_state.history.append({"role": "assistant", "content": text, "sources": cites})
