from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pretty_midi
import os
from pathlib import Path
import uuid
from typing import Optional
import json

from app.core.generator import AMTBackingGenerator

router = APIRouter()
generator = AMTBackingGenerator()

# Models for request validation
class GenerationRequest(BaseModel):
    num_bars: int = 8
    tempo: int = 120
    key: Optional[int] = 60

# Store for active generation tasks
active_tasks = {}

@router.post("/backing")
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

        # Save audio file
        audio_path = output_dir / "backing.wav"
        audio_data = generator.midi_to_audio(backing_track)
        with open(audio_path, 'wb') as f:
            f.write(audio_data.tobytes())

        return {
            "generation_id": generation_id,
            "midi_url": f"/static/{generation_id}/backing.mid",
            "audio_url": f"/static/{generation_id}/backing.wav",
            "status": "completed"
        }

    except Exception as e:
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