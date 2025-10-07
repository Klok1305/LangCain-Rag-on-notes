RAG Chatbot — краткая инструкция

Что установить
Python 3.10–3.12
Ollama
   Модели для Ollama: 
   
    ```ollama pull gemma3n:e2b
    ollama pull nomic-embed-text```

Cтруктура папок (создать, если нет)
   data/pdf   — сюда класть PDF
   data/txt   — сюда класть .txt или .md
   chroma     — сюда будет записан индекс
   src        — тут лежат ingest.py и rag.py

   Windows:
     New-Item -ItemType Directory -Force data\pdf, data\txt, chroma
   macOS/Linux:
     mkdir -p data/pdf data/txt chroma

Виртуальное окружение и зависимости
   Windows:
    ```
     py -3 -m venv .venv
     .\.venv\Scripts\Activate.ps1
   macOS/Linux:
     python3 -m venv .venv
     source .venv/bin/activate
    ```

   Общее:
     ```python -m pip install -U pip
     python -m pip install -U "langchain>=0.2.16" "langchain-core>=0.2.38" "langchain-community>=0.2.16" "langchain-ollama>=0.1.0" "langchain-chroma>=0.1.0" "chromadb>=0.5.5" ddgs pypdf Pillow```

Запуск
   4.1 В одном окне:
   ollama serve
   4.2 В другом окне, из корня проекта:
      python .\src\ingest.py
      python .\src\rag.py


Полезно знать
• Документы кладём в data/pdf и data/txt
• Если не находит ответы, снизь RELEVANCE_THRESHOLD в rag.py до 0.6 и подними RETRIEVER_K до 10
• При массовых изменениях данных удали папку chroma и заново запусти ingest.py
