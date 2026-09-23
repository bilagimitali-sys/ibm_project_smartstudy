# 🎓 StudyMate AI — Smart Study Generator Agent

> An AI-powered study companion that turns your PDFs and notes into summaries, exam questions, flashcards, MCQ quizzes, and personalised study plans — all in one Streamlit app.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA3-orange)
![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📁 **Document Upload** | Upload PDFs, TXT, or Markdown notes; auto-indexed into FAISS |
| 📄 **AI Summary** | Structured summaries with key concepts and connections |
| ❓ **Exam Questions** | Targeted Q&A with model answers at adjustable difficulty |
| 🃏 **Flashcards** | Interactive flip-cards for active recall practice |
| 📝 **MCQ Quiz** | Multiple-choice quiz with instant scoring and answer review |
| 📅 **Study Plan** | Personalised day-by-day schedule based on your exam date |
| 📊 **Dashboard** | Live session stats, quiz score history, and activity log |

---

## 🏗️ Architecture

```
studymate_ai/
├── app.py                     ← Streamlit entry point & page router
├── src/
│   ├── rag/
│   │   ├── document_processor.py  ← PDF/TXT ingestion & chunking
│   │   ├── vector_store.py        ← FAISS index build/load/query
│   │   └── llm_client.py          ← Groq API wrapper (LLaMA3)
│   └── features/
│       ├── summary.py             ← AI Summary generation
│       ├── exam_questions.py      ← Exam question generation
│       ├── flashcards.py          ← Flashcard generation & parsing
│       ├── mcq_quiz.py            ← MCQ quiz generation & parsing
│       ├── study_plan.py          ← Personalised study plan
│       └── dashboard.py           ← Progress dashboard & tracking
├── requirements.txt
├── .env.example
└── README.md
```

**RAG Pipeline:**
1. Upload → chunk with `RecursiveCharacterTextSplitter` (1 000 tokens, 150 overlap)
2. Embed chunks with `all-MiniLM-L6-v2` (local, no extra API key)
3. Store in FAISS; persist to `faiss_index/`
4. On each feature request → retrieve top-5 relevant chunks → inject into prompt → call Groq LLaMA3

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/bilagimitali-sys/ibm_project_smartstudy.git
cd ibm_project_smartstudy
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure your API key
```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

Get a free Groq API key at → https://console.groq.com

### 5. Run the app
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | Groq API key (LLaMA3 inference) |
| `IBM_WATSON_API_KEY` | Optional | IBM watsonx.ai API key |
| `IBM_WATSON_PROJECT_ID` | Optional | IBM watsonx project ID |
| `IBM_WATSON_URL` | Optional | watsonx endpoint URL |

> ⚠️ **Never commit your `.env` file.** It is listed in `.gitignore`.

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io) 1.35
- **LLM**: [Groq](https://groq.com) (LLaMA3-8B / LLaMA3-70B)
- **RAG framework**: [LangChain](https://langchain.com) 0.2
- **Vector DB**: [FAISS](https://github.com/facebookresearch/faiss)
- **Embeddings**: [sentence-transformers](https://sbert.net) `all-MiniLM-L6-v2`
- **PDF parsing**: [pypdf](https://pypdf.readthedocs.io)
- **Charts**: [Plotly](https://plotly.com/python/)

---

## 📸 Demo Flow

1. Upload your lecture PDF → documents are chunked and indexed
2. Go to **Summary** → get a structured overview in seconds
3. Go to **Exam Questions** → pick difficulty and get 10 practice questions
4. Go to **Flashcards** → flip through 15 auto-generated cards
5. Go to **MCQ Quiz** → take a quiz and see your score instantly
6. Go to **Study Plan** → enter your exam date and get a full schedule
7. Go to **Dashboard** → see your activity, quiz scores, and progress

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
