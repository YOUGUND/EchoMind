<div align="center">

# EchoMind

### An observable, evaluable multi-agent customer-service runtime

**Understand · Route · Act · Pack context · Evaluate**

</div>

## Overview

EchoMind turns one user request into an engineering pipeline:

**intent → primary/supporting agent routing → skills/knowledge → tools → context packing → compose → trace/eval**

It is built for internship / portfolio demos of agent systems with clear role boundaries, recoverable runtime state, and reproducible quality gates.

## Stack

- Frontend: Vue 3 + Vite
- Backend: FastAPI + LangGraph
- Data: PostgreSQL + Milvus (JWT multi-user)
- Models: SiliconFlow-compatible gateway

## Quick start

```powershell
docker compose -f docker-compose.multi.yml up -d
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-free.txt
Copy-Item .env.example .env.local
cd ..\frontend
npm ci
cd ..
python 启动开发.py
```

See `docs/multi-user.md` for JWT + storage details.

## Author

**YOUGUND** · https://github.com/YOUGUND
