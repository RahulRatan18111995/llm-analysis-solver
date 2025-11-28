# LLM Analysis Quiz Solver

This repository contains my IITM LLM-Analysis Project submission.

## Features
- FastAPI endpoint `/receive_request`
- Secret verification
- Playwright-based DOM rendering
- CSV / PDF / XLSX parsing
- JSON extraction from <pre>, atob(Base64), loose JSON
- AiPipe fallback LLM
- Automatic solver that follows IITM quiz URLs

## Deployment
Deployed on Render using:
- `uvicorn app:app`
- Playwright chromium installed on build
