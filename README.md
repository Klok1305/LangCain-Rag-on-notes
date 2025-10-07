    RAG Chatbot — краткая инструкция

    Что установить:
      • Python 3.10–3.12
      • Ollama
      • Модели для Ollama:
          ollama pull gemma3n:e2b
          ollama pull nomic-embed-text

    Структура папок:
      data/pdf   — сюда класть PDF
      data/txt   — сюда класть .txt или .md
      chroma     — сюда будет записан индекс
      src        — тут лежат ingest.py и rag.py

      Windows:
        New-Item -ItemType Directory -Force data\pdf, data\txt, chroma
      macOS/Linux:
        mkdir -p data/pdf data/txt chroma

    Виртуальное окружение:
      Windows:
        py -3 -m venv .venv
        .\.venv\Scripts\Activate.ps1
      macOS/Linux:
        python3 -m venv .venv
        source .venv/bin/activate

    Установка зависимостей:
      python -m pip install -U pip
      python -m pip
