"""Vercel serverless entry point.

Vercel's Python runtime discovers this file and serves the FastAPI app
as a serverless function. All requests matching /api/* are routed here
via vercel.json rewrites.

Local development: uvicorn web.app:app --reload
"""

from web.app import app
