#!/bin/bash
# FastAPI 서버 실행 스크립트

echo "Starting FastAPI RAG Chatbot Server..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
