# Retrieval-Augmented Generation (RAG) Application – Dockerized

This repository contains a Retrieval-Augmented Generation (RAG) application.
The project covers data cleaning and enrichment, RAG pipeline design and explanation,
and a fully dockerized Streamlit application for easy testing and reproducibility.

---

## Repository Overview

The repository is organized as follows:

```
.
├── data_manipulation/
├── rag_notebook/
├── app/
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Data Manipulation and Enrichment

All steps related to **data manipulation** are documented in the
`data_manipulation/` folder.

The workflow followed is:

1. **Dataset Cleaning**  
   The original dataset was cleaned to remove noise, fix formatting issues,
   and ensure consistency.

2. **Dataset Enrichment**  
   After cleaning, the dataset was enriched by improving and expanding
   textual descriptions using **OpenRouter**.  
   This step was performed to enhance semantic quality and improve retrieval
   performance in the RAG system.

Please refer to the `data_manipulation/` folder for scripts and notebooks
detailing this process.

---

## RAG Pipeline Explanation

To understand the **RAG pipeline design and logic**, refer to the
`rag_notebook/` folder.

This folder contains notebooks that explain:
- Document chunking strategy
- Embedding generation
- Vector store usage
- Retrieval mechanism
- How retrieved context is combined with the user query and passed to the LLM

These notebooks provide a detailed, step-by-step explanation of the complete
RAG pipeline.

---

## Running and Testing the RAG System (Dockerized)

The application is fully containerized using Docker, allowing the system to be
run without any local Python environment setup.

### Prerequisites
- Docker installed on your system

---

### Step 1: Clone the Repository
```bash
git clone https://github.com/alihamzeh1997/rag-app.git
cd rag-app
```

---

### Step 2: Set Environment Variables
Create a `.env` file in the project root (based on `.env.example`) and add:

```bash
OPENAI_API_KEY=your_api_key_here
```

---

### Step 3: Build the Docker Image
```bash
docker build -t rag-app .
```

---

### Step 4: Run the Docker Container
```bash
docker run --env-file .env -p 8501:8501 rag-app
```

---

### Step 5: Access the Application
Open a web browser and navigate to:

```
http://localhost:8501
```

The Streamlit interface will load, and the RAG system can be tested end-to-end.

---

## Notes
- Dependencies are locked using `uv.lock` to ensure reproducibility.
- No API keys or sensitive information are included in the repository or Docker image.
- The application runs entirely inside the Docker container.
