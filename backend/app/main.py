from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uvicorn

# Create necessary directories
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

app = FastAPI(
    title="AMT Backing Generator API",
    description="API for generating AI-powered backing tracks and lead melodies",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Import and include routers
from app.routes import generator
app.include_router(generator.router, prefix="/api", tags=["generator"])

@app.get("/")
async def root():
    return {"message": "AMT Backing Generator API"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 