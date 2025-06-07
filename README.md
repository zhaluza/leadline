# AMT Backing Track Generator

This project uses the Anticipatory Music Transformer (AMT) to generate backing tracks and lead melodies. It's designed to be a simple tool for guitarists to create backing tracks with AI-generated melodies.

## Features

- Generate backing tracks with:
  - Piano chords (C - Am - F - G progression)
  - Bass line
  - Drum pattern
- Generate AI lead melodies using AMT
- Preview audio directly in Jupyter notebooks
- Save all tracks as MIDI files

## Installation

1. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

You can use this code in two ways:

1. Run as a script:

```bash
python amt_backing_generator.py
```

2. Use in a Jupyter notebook:

```python
from amt_backing_generator import AMTBackingGenerator

# Initialize the generator
generator = AMTBackingGenerator()

# Generate a backing track
backing_track = generator.create_backing_track()
backing_audio = generator.save_and_preview(backing_track, "backing_track.mid")

# Generate a lead melody
lead_melody = generator.generate_lead_melody(backing_track)
if lead_melody:
    lead_audio = generator.save_and_preview(lead_melody, "lead_melody.mid")
```

## Output Files

All generated MIDI files are saved in the `output` directory:

- `backing_track.mid`: The backing track with chords, bass, and drums
- `lead_melody.mid`: The AI-generated lead melody
- `combined_track.mid`: The backing track with the lead melody

## Notes

- The backing track uses a simple C - Am - F - G chord progression
- The lead melody is generated using the AMT model and filtered to stay in a musical range
- All tracks are 16 seconds long by default
- Audio previews are generated at 44.1kHz sample rate
