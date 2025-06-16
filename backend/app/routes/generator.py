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
    print("✅ Successfully initialized clean generator")
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
    chord_progression: Optional[list[str]] = None  # e.g., ["C", "Am", "F", "G"]

class ChordBackingRequest(BaseModel):
    num_bars: int = 8
    tempo: int = 120
    key: str = "C"  # e.g., "C", "G", "F#", etc.
    chord_progression: list[str]  # e.g., ["C", "Am", "F", "G"]

class SeedNote(BaseModel):
    pitch: int  # MIDI note number (0-127)
    start: float  # Start time in beats (0.0, 1.0, 2.0, etc.)
    duration: float  # Duration in beats (0.25, 0.5, 1.0, etc.)

class LeadMelodyRequest(BaseModel):
    num_bars: int = 8
    tempo: int = 120
    key: str = "C"
    seed_notes: Optional[list[SeedNote]] = None  # User-provided seed notes for first measure

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

@router.post("/backing/chords")
@retry_on_error(max_retries=3, delay=2)
async def generate_backing_with_chords(request: ChordBackingRequest):
    """Generate a backing track with custom chord progression and return both MIDI and audio files."""
    try:
        # Generate unique ID for this generation
        generation_id = str(uuid.uuid4())
        output_dir = Path("static") / generation_id
        output_dir.mkdir(exist_ok=True)

        # Generate backing track with custom chords
        backing_track = generator.create_backing_track_with_chords(
            chord_progression=request.chord_progression,
            key=request.key,
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
            "status": "completed",
            "chord_progression": request.chord_progression,
            "key": request.key
        }

    except Exception as e:
        print(f"Error generating backing track with chords: {e}")
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

@router.post("/lead/{generation_id}")
@retry_on_error(max_retries=3, delay=2)
async def generate_lead(generation_id: str, request: GenerationRequest):
    """Generate a lead melody for an existing backing track and return combined track as well."""
    try:
        print(f"[DEBUG] Starting lead melody generation for {generation_id}")
        print(f"[DEBUG] Request params: num_bars={request.num_bars}, tempo={request.tempo}, key={request.key}")
        
        # Verify backing track exists
        backing_dir = Path("static") / generation_id
        backing_midi = backing_dir / "backing.mid"
        print(f"[DEBUG] Looking for backing track at {backing_midi}")
        if not backing_midi.exists():
            print(f"[DEBUG] Backing track not found at {backing_midi}")
            raise HTTPException(status_code=404, detail="Backing track not found")
        print(f"[DEBUG] Found backing track at {backing_midi}")

        # Load the backing track
        print("[DEBUG] Loading backing track MIDI file")
        try:
            backing_track = pretty_midi.PrettyMIDI(str(backing_midi))
            print(f"[DEBUG] Loaded backing track with {len(backing_track.instruments)} instruments")
            for i, inst in enumerate(backing_track.instruments):
                print(f"[DEBUG] Instrument {i}: program={inst.program}, is_drum={inst.is_drum}, notes={len(inst.notes)}")
        except Exception as e:
            print(f"[DEBUG] Error loading backing track: {e}")
            import traceback
            print(f"[DEBUG] Backing track load traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Failed to load backing track: {str(e)}")
        
        # Generate lead melody
        print("[DEBUG] Calling generator.generate_lead_melody")
        try:
            lead_melody = generator.generate_lead_melody(
                backing_track,
                num_bars=request.num_bars,
                tempo=request.tempo
            )
            print("[DEBUG] Lead melody generation completed")
        except Exception as e:
            print(f"[DEBUG] Error in lead melody generation: {e}")
            import traceback
            print(f"[DEBUG] Lead melody generation traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Failed to generate lead melody: {str(e)}")
        
        if lead_melody is None:
            print("[DEBUG] Lead melody generation returned None")
            raise HTTPException(status_code=500, detail="Failed to generate lead melody")

        # Save lead MIDI file
        print("[DEBUG] Saving lead MIDI file")
        try:
            midi_path = backing_dir / "lead.mid"
            lead_melody.write(str(midi_path))
            print(f"[DEBUG] Saved lead MIDI file to {midi_path}")
        except Exception as e:
            print(f"[DEBUG] Error saving lead MIDI: {e}")
            import traceback
            print(f"[DEBUG] Lead MIDI save traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Failed to save lead MIDI: {str(e)}")

        # Save lead audio file
        audio_path = backing_dir / "lead.wav"
        print(f"Generating lead audio file at {audio_path}")
        audio_data = generator.midi_to_audio(lead_melody)
        if audio_data is None or len(audio_data) == 0:
            raise ValueError("Generated audio data is empty")
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_data = (audio_data * 32767).astype(np.int16)
        wavfile.write(str(audio_path), 44100, audio_data)

        # Create combined track
        combined_midi = pretty_midi.PrettyMIDI(initial_tempo=request.tempo)
        
        # Add backing track instruments
        for instrument in backing_track.instruments:
            combined_midi.instruments.append(instrument)
        
        # Add lead melody
        for instrument in lead_melody.instruments:
            combined_midi.instruments.append(instrument)

        # Save combined MIDI file
        combined_midi_path = backing_dir / "combined.mid"
        combined_midi.write(str(combined_midi_path))
        print(f"Saved combined MIDI file to {combined_midi_path}")

        # Save combined audio file
        combined_audio_path = backing_dir / "combined.wav"
        print(f"Generating combined audio file at {combined_audio_path}")
        combined_audio_data = generator.midi_to_audio(combined_midi)
        if combined_audio_data is None or len(combined_audio_data) == 0:
            raise ValueError("Generated combined audio data is empty")
        if np.max(np.abs(combined_audio_data)) > 1.0:
            combined_audio_data = np.clip(combined_audio_data, -1.0, 1.0)
        combined_audio_data = (combined_audio_data * 32767).astype(np.int16)
        wavfile.write(str(combined_audio_path), 44100, combined_audio_data)

        return {
            "generation_id": generation_id,
            "midi_url": f"/static/{generation_id}/lead.mid",
            "audio_url": f"/static/{generation_id}/lead.wav",
            "combined_midi_url": f"/static/{generation_id}/combined.mid",
            "combined_audio_url": f"/static/{generation_id}/combined.wav",
            "status": "completed"
        }

    except Exception as e:
        print(f"Error generating lead melody: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/lead/{generation_id}/custom")
@retry_on_error(max_retries=3, delay=2)
async def generate_lead_with_seed(generation_id: str, request: LeadMelodyRequest):
    """Generate a lead melody with custom seed notes for an existing backing track."""
    try:
        # Verify backing track exists
        backing_dir = Path("static") / generation_id
        backing_midi = backing_dir / "backing.mid"
        if not backing_midi.exists():
            raise HTTPException(status_code=404, detail="Backing track not found")

        # Load the backing track
        backing_track = pretty_midi.PrettyMIDI(str(backing_midi))
        
        # Create seed melody from user notes if provided
        seed_melody = None
        if request.seed_notes:
            # Convert SeedNote objects to dictionaries
            seed_notes_data = [
                {
                    'pitch': note.pitch,
                    'start': note.start,
                    'duration': note.duration
                }
                for note in request.seed_notes
            ]
            seed_melody = generator.create_seed_melody_from_notes(
                seed_notes_data,
                key=request.key,
                num_bars=request.num_bars,
                tempo=request.tempo
            )
            print(f"Created custom seed melody with {len(request.seed_notes)} notes")
        
        # Generate lead melody with custom seed
        lead_melody = generator.generate_lead_melody_with_seed(
            backing_track,
            seed_melody=seed_melody,
            num_bars=request.num_bars,
            tempo=request.tempo
        )
        
        if lead_melody is None:
            raise HTTPException(status_code=500, detail="Failed to generate lead melody")

        # Save lead MIDI file
        midi_path = backing_dir / "lead.mid"
        lead_melody.write(str(midi_path))
        print(f"Saved lead MIDI file to {midi_path}")

        # Save lead audio file
        audio_path = backing_dir / "lead.wav"
        print(f"Generating lead audio file at {audio_path}")
        audio_data = generator.midi_to_audio(lead_melody)
        if audio_data is None or len(audio_data) == 0:
            raise ValueError("Generated audio data is empty")
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_data = (audio_data * 32767).astype(np.int16)
        wavfile.write(str(audio_path), 44100, audio_data)

        # Create combined track
        combined_midi = pretty_midi.PrettyMIDI(initial_tempo=request.tempo)
        
        # Add backing track instruments
        for instrument in backing_track.instruments:
            combined_midi.instruments.append(instrument)
        
        # Add lead melody
        for instrument in lead_melody.instruments:
            combined_midi.instruments.append(instrument)

        # Save combined MIDI file
        combined_midi_path = backing_dir / "combined.mid"
        combined_midi.write(str(combined_midi_path))
        print(f"Saved combined MIDI file to {combined_midi_path}")

        # Save combined audio file
        combined_audio_path = backing_dir / "combined.wav"
        print(f"Generating combined audio file at {combined_audio_path}")
        combined_audio_data = generator.midi_to_audio(combined_midi)
        if combined_audio_data is None or len(combined_audio_data) == 0:
            raise ValueError("Generated combined audio data is empty")
        if np.max(np.abs(combined_audio_data)) > 1.0:
            combined_audio_data = np.clip(combined_audio_data, -1.0, 1.0)
        combined_audio_data = (combined_audio_data * 32767).astype(np.int16)
        wavfile.write(str(combined_audio_path), 44100, combined_audio_data)

        return {
            "generation_id": generation_id,
            "midi_url": f"/static/{generation_id}/lead.mid",
            "audio_url": f"/static/{generation_id}/lead.wav",
            "combined_midi_url": f"/static/{generation_id}/combined.mid",
            "combined_audio_url": f"/static/{generation_id}/combined.wav",
            "status": "completed",
            "seed_notes_used": len(request.seed_notes) if request.seed_notes else 0
        }

    except Exception as e:
        print(f"Error generating lead melody with seed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/lead/{generation_id}/preview-seed")
async def preview_seed_melody(generation_id: str, request: GenerationRequest):
    """Generate and preview the seed melody that will be used for lead generation."""
    try:
        # Verify backing track exists and extract its tempo
        backing_dir = Path("static") / generation_id
        backing_midi = backing_dir / "backing.mid"
        if not backing_midi.exists():
            raise HTTPException(status_code=404, detail="Backing track not found")

        # Load the backing track to extract its tempo
        backing_track = pretty_midi.PrettyMIDI(str(backing_midi))
        
        # Extract tempo from backing track
        backing_tempo = request.tempo  # Default to requested tempo
        try:
            tempo_times, tempo_values = backing_track.get_tempo_changes()
            if len(tempo_values) > 0:
                backing_tempo = int(tempo_values[0])
                print(f"[DEBUG] Extracted tempo from backing track for seed preview: {backing_tempo} BPM")
            else:
                print(f"[DEBUG] Using requested tempo for seed preview: {request.tempo} BPM")
        except Exception as e:
            print(f"[DEBUG] Could not extract tempo from backing track: {e}")
            print(f"[DEBUG] Using requested tempo for seed preview: {request.tempo} BPM")
        
        # Generate seed melody for just 1 bar (this is what's actually used as seed)
        # Use the backing track's tempo instead of the requested tempo
        seed_melody = generator.create_seed_melody(
            key=request.key,
            num_bars=1,  # Only 1 bar as seed
            tempo=backing_tempo  # Use backing track tempo
        )
        
        # Save seed melody files
        output_dir = Path("static") / generation_id
        output_dir.mkdir(exist_ok=True)
        
        # Save MIDI file
        midi_path = output_dir / "preview_seed.mid"
        seed_melody.write(str(midi_path))
        
        # Save audio file
        audio_path = output_dir / "preview_seed.wav"
        audio_data = generator.audio_converter.midi_to_audio_no_adjustment(seed_melody)
        if audio_data is None or len(audio_data) == 0:
            raise ValueError("Generated audio data is empty")
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_data = (audio_data * 32767).astype(np.int16)
        wavfile.write(str(audio_path), 44100, audio_data)
        
        return {
            "generation_id": generation_id,
            "midi_url": f"/static/{generation_id}/preview_seed.mid",
            "audio_url": f"/static/{generation_id}/preview_seed.wav",
            "status": "completed",
            "tempo_used": backing_tempo  # Return the actual tempo used
        }
        
    except Exception as e:
        print(f"Error generating seed preview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files/{generation_id}/{filename}")
async def get_file(generation_id: str, filename: str):
    """Get a generated file by ID and filename."""
    file_path = Path("static") / generation_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path) 