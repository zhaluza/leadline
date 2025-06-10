from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uvicorn

# Create necessary directories
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

app = FastAPI(
    title="LeadLine API",
    description="AI-powered backing tracks and lead melody generator",
    version="1.0.0"
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
    return {"message": "Welcome to LeadLine - AI Backing Tracks & Melodies"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 