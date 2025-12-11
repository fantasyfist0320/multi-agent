🚀 Setup & Run Instructions

This guide explains how to install dependencies, prepare the environment, ingest policy PDFs, and run the multi-agent insurance assistant locally.

1. 📦 Clone the Repository

git clone https://github.com/fantasyfist0320/multi-agent.git
cd multi-agent

2. 🐍 Create & Activate Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

3. 📥 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. 🔑 Configure Environment Variables

5. 📚 Build the Policy Vector Index (RAG)

```bash
python -m app.tools.policy_retriever
```
6. ▶️ Run Manual Test

```bash
python -m tests.manual_test
```