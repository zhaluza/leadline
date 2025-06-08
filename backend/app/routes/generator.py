from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pretty_midi
import os
from pathlib import Path
import uuid
from typing import Optional
import json
import time
from functools import wraps
import numpy as np
from scipy.io import wavfile

from app.core.generator import AMTBackingGenerator

router = APIRouter()

# Get the project root directory (three levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SOUNDFONT_PATH = PROJECT_ROOT / "soundfonts" / "MuseScore_General.sf3"

def retry_on_error(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1} failed: {str(e)}")
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
            raise HTTPException(
                status_code=500,
                detail=f"Failed after {max_retries} attempts. Last error: {str(last_error)}"
            )
        return wrapper
    return decorator

# Initialize generator with soundfont
try:
    print(f"Initializing generator with soundfont: {SOUNDFONT_PATH}")
    if not SOUNDFONT_PATH.exists():
        raise FileNotFoundError(f"Soundfont not found at {SOUNDFONT_PATH}")
    generator = AMTBackingGenerator(soundfont_path=str(SOUNDFONT_PATH))
except Exception as e:
    print(f"Error initializing generator: {e}")
    raise HTTPException(
        status_code=500,
        detail=f"Failed to initialize generator: {str(e)}"
    )

# Models for request validation
class GenerationRequest(BaseModel):
    num_bars: int = 8
    tempo: int = 120
    key: Optional[int] = 60

# Store for active generation tasks
active_tasks = {}

@router.post("/backing")
@retry_on_error(max_retries=3, delay=2)
async def generate_backing(request: GenerationRequest):
    """Generate a backing track and return both MIDI and audio files."""
    try:
        # Generate unique ID for this generation
        generation_id = str(uuid.uuid4())
        output_dir = Path("static") / generation_id
        output_dir.mkdir(exist_ok=True)

        # Generate backing track
        backing_track = generator.create_backing_track(
            num_bars=request.num_bars,
            tempo=request.tempo
        )

        # Save MIDI file
        midi_path = output_dir / "backing.mid"
        backing_track.write(str(midi_path))
        print(f"Saved MIDI file to {midi_path}")

        # Save audio file
        audio_path = output_dir / "backing.wav"
        print(f"Generating audio file at {audio_path}")
        audio_data = generator.midi_to_audio(backing_track)
        
        # Verify audio data
        if audio_data is None or len(audio_data) == 0:
            raise ValueError("Generated audio data is empty")
        
        print(f"Audio data shape: {audio_data.shape}, dtype: {audio_data.dtype}")
        print(f"Audio data min: {np.min(audio_data)}, max: {np.max(audio_data)}")
        
        # Ensure audio data is in the correct range (-1 to 1)
        if np.max(np.abs(audio_data)) > 1.0:
            print("Normalizing audio data to [-1, 1] range")
            audio_data = np.clip(audio_data, -1.0, 1.0)
        
        # Convert to 16-bit PCM
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Write WAV file with proper headers using scipy
        sample_rate = 44100  # Standard sample rate
        wavfile.write(str(audio_path), sample_rate, audio_data)
        
        # Verify file was written
        if not audio_path.exists():
            raise FileNotFoundError(f"Failed to write audio file to {audio_path}")
        
        file_size = audio_path.stat().st_size
        print(f"Audio file written: {audio_path}, size: {file_size} bytes")

        return {
            "generation_id": generation_id,
            "midi_url": f"/static/{generation_id}/backing.mid",
            "audio_url": f"/static/{generation_id}/backing.wav",
            "status": "completed"
        }

    except Exception as e:
        print(f"Error generating backing track: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/seed")
async def generate_seed(request: GenerationRequest):
    """Generate a seed melody and return both MIDI and audio files."""
    try:
        generation_id = str(uuid.uuid4())
        output_dir = Path("static") / generation_id
        output_dir.mkdir(exist_ok=True)

        # Generate seed melody
        seed_melody = generator.create_seed_melody(
            key=request.key,
            num_bars=request.num_bars,
            tempo=request.tempo
        )

        # Save MIDI file
        midi_path = output_dir / "seed.mid"
        seed_melody.write(str(midi_path))

        # Save audio file
        audio_path = output_dir / "seed.wav"
        audio_data = generator.midi_to_audio(seed_melody)
        with open(audio_path, 'wb') as f:
            f.write(audio_data.tobytes())

        return {
            "generation_id": generation_id,
            "midi_url": f"/static/{generation_id}/seed.mid",
            "audio_url": f"/static/{generation_id}/seed.wav",
            "status": "completed"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/lead")
async def generate_lead(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Generate a lead melody using the AMT model."""
    try:
        generation_id = str(uuid.uuid4())
        output_dir = Path("static") / generation_id
        output_dir.mkdir(exist_ok=True)

        # Create backing track for conditioning
        backing_track = generator.create_backing_track(
            num_bars=request.num_bars,
            tempo=request.tempo
        )

        # Generate lead melody
        lead_melody = generator.generate_lead_melody(
            backing_track,
            num_bars=request.num_bars,
            tempo=request.tempo
        )

        if lead_melody is None:
            raise HTTPException(status_code=500, detail="Failed to generate lead melody")

        # Save MIDI file
        midi_path = output_dir / "lead.mid"
        lead_melody.write(str(midi_path))

        # Save audio file
        audio_path = output_dir / "lead.wav"
        audio_data = generator.midi_to_audio(lead_melody)
        with open(audio_path, 'wb') as f:
            f.write(audio_data.tobytes())

        # Save combined track
        combined = pretty_midi.PrettyMIDI()
        for instrument in backing_track.instruments:
            combined.instruments.append(instrument)
        for instrument in lead_melody.instruments:
            combined.instruments.append(instrument)

        combined_midi_path = output_dir / "combined.mid"
        combined.write(str(combined_midi_path))

        combined_audio_path = output_dir / "combined.wav"
        combined_audio = generator.midi_to_audio(combined)
        with open(combined_audio_path, 'wb') as f:
            f.write(combined_audio.tobytes())

        return {
            "generation_id": generation_id,
            "midi_url": f"/static/{generation_id}/lead.mid",
            "audio_url": f"/static/{generation_id}/lead.wav",
            "combined_midi_url": f"/static/{generation_id}/combined.mid",
            "combined_audio_url": f"/static/{generation_id}/combined.wav",
            "status": "completed"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files/{generation_id}/{filename}")
async def get_file(generation_id: str, filename: str):
    """Get a generated file by ID and filename."""
    file_path = Path("static") / generation_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path) 