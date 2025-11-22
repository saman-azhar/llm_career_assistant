from fastapi import FastAPI
from career_assistant.api.endpoints import match, health, metrics, collections

app = FastAPI(title="Career Assistant API")

# Mount routers
app.include_router(match.router)
app.include_router(health.router)
app.include_router(metrics.router)
app.include_router(collections.router)


