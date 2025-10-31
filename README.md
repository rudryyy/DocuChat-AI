 DocuChat AI · Multimodal RAG (Text · Image · Audio)

A Streamlit-based Multimodal Retrieval-Augmented Generation (RAG) system that indexes text, images, and audio into a shared CLIP embedding space.
It retrieves contextually relevant data and generates grounded, cited answers—all without external FFmpeg dependencies.

> ⚠️ For research and educational use only · Not for commercial deployment.

-----

🚀 Features

  - 🔍 Unified multimodal search — query by text, image, or audio.
  - 🧠 CLIP + FAISS backbone for vector similarity search across modalities.
  - 🔊 Whisper integration for automatic speech-to-text on uploaded or YouTube audio.
  - 📄 Document support: PDF, TXT, DOCX, web pages, and YouTube captions.
  - 🎥 FFmpeg-free audio decoding: uses PyAV and SoundFile only (works in restricted environments).
  - 💡 Grounded answers with citations via Llama 3 (Ollama LLM backend).
  - 🌐 Robust YouTube pipeline with format fallback and duration validation.

-----

# 🧰 Tech Stack

| Category | Technology |
|-----------|-------------|
| Framework | Streamlit |
| Embeddings | Sentence-Transformers (clip-ViT-B-32) |
| Indexing | FAISS (Inner Product search + L2 normalization) |
| LLM | Ollama (Llama 3 models) |
| Audio Decode | PyAV + SoundFile (no external FFmpeg) |
| Speech to Text | Whisper |
| Document Parsing | LangChain Loaders (PyPDF, Text, Docx2txt) |
| Video/Audio Input | yt-dlp (YouTube download + validation) |

-----

