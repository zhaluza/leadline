import sys
import os

# Add the current directory to Python path so we can import our app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    try:
        # Try to import uvicorn
        import uvicorn
        print("Starting server with uvicorn...")
        uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
    except ImportError:
        print("Uvicorn not found, using hypercorn...")
        try:
            from hypercorn.asyncio import serve
            from hypercorn.config import Config
            from app.main import app
            import asyncio
            
            config = Config()
            config.bind = ["0.0.0.0:8000"]
            config.use_reloader = True
            
            print("Starting server with hypercorn...")
            asyncio.run(serve(app, config))
        except ImportError:
            print("Neither uvicorn nor hypercorn found. Please install one of them:")
            print("pip install uvicorn")
            print("or")
            print("pip install hypercorn")
            sys.exit(1) 