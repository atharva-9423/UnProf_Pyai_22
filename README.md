<div align="center">

# 🧠 Day 22 — Conversational RAG
### Context-Aware Q&A with Memory

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-green?style=for-the-badge&logo=langchain&logoColor=white)](https://python.langchain.com/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-1.5%20Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://aistudio.google.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-red?style=for-the-badge&logo=facebook&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Embeddings-ff9d00?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/)

*An upgraded Retrieval-Augmented Generation (RAG) Chatbot featuring Conversation Memory and Context-Aware follow-up handling.*

</div>

---

## 📖 Overview

This project upgrades our existing Document-based RAG CLI by introducing **Conversation Memory**. The chatbot now "remembers" previous interactions in the session. It automatically rewrites follow-up questions (e.g., "What is it?") into standalone questions (e.g., "What is Machine Learning?") before retrieving documents from the vector database, enabling seamless multi-turn conversations!

## ✨ Key Features

- **💬 Conversation Memory:** Manages a history of user questions and AI answers.
- **🔄 Contextual Question Reformulation:** Dynamically rewrites follow-up questions to resolve pronouns and context based on the chat history.
- **🧠 Advanced RAG Orchestration:** Two-step LCEL (LangChain Expression Language) pipeline for question reformulation and document-grounded answering.
- **📄 Document Loading & Embeddings:** Ingests local `.txt` files into a high-performance FAISS index.

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python** | Core Language |
| **LangChain (LCEL)** | RAG Pipeline Orchestration & Memory Handling |
| **FAISS** | Fast Vector Database Search |
| **Sentence Transformers** | Hugging Face Embeddings generation |
| **Google Gemini API** | Advanced LLM Generation |

## 📁 Project Structure

```text
UnProf_Pyai_22/
├── main.py               # Conversational RAG application utilizing LangChain LCEL
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── documents/
    └── sample_notes.txt  # Sample document for testing RAG
```

## 🚀 Setup & Usage

### 1. Install Dependencies
```bash
pip install -U -r requirements.txt
```

### 2. Set the Gemini API Key
Provide your valid Google Gemini API key as an environment variable (Must start with `AIza`):

**Windows (PowerShell)**
```powershell
$env:GEMINI_API_KEY="AIza..."
```
**Linux / macOS**
```bash
export GEMINI_API_KEY="AIza..."
```

### 3. Run the Application
Execute the Conversational RAG CLI:
```bash
python main.py
```

### 4. Test Conversational Memory!
Ask an initial question:
> **You:** "What is AI?"

Then, ask a follow-up question that relies on memory:
> **You:** "When was a major milestone reached for it?"

Notice how the AI understands that "it" refers to AI, reformulates the question, retrieves the correct context, and provides the answer! Type `exit` to quit.

---
<div align="center">
<i>Built for the 100 Days of Code challenge. Phase 3 - LLM & RAG Core.</i>
</div>
