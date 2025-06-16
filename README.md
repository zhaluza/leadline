# LeadLine - AI-Powered Backing Tracks & Melodies

LeadLine is a full-stack web application that generates AI-powered backing tracks and lead melodies for guitarists. Built with React frontend and FastAPI backend, it uses Anticipatory Music Transformers (AMT) to create musically coherent practice material.

## 🎵 Features

### Backing Track Generation

- **Custom Chord Progressions**: Create backing tracks with any chord progression (C-Am-F-G, ii-V-I, etc.)
- **Multiple Instruments**: Piano chords, bass lines, and drum patterns
- **Key & Tempo Control**: Adjust key signature and tempo (60-200 BPM)
- **Flexible Length**: Generate 4-32 bar backing tracks

### AI Lead Melody Generation

- **AMT-Powered**: Uses Stanford's music-medium-800k model for intelligent melody generation
- **Seed Melody Options**:
  - **Preset**: Algorithmically generated seed melodies
  - **Custom**: User-defined seed notes for personalized direction
- **Musical Intelligence**: Generates melodies that fit the harmonic context
- **Fallback System**: Algorithmic generation if AI fails

### Audio & Visualization

- **High-Quality Audio**: FluidSynth with professional soundfonts
- **MIDI Visualization**: Interactive piano roll view
- **Real-time Playback**: Built-in audio players for all tracks
- **Combined Output**: Backing track + lead melody in one file

## 🏗️ Architecture

### Frontend (React + TypeScript)

- **React 18** with TypeScript for type safety
- **Vite** for fast development and building
- **Tailwind CSS** for modern, responsive UI
- **Interactive Components**: BackingTrack, LeadMelody, AudioPlayer, MidiVisualizer

### Backend (FastAPI + Python)

- **FastAPI** for REST API endpoints
- **AMT Model**: Stanford CRFM music-medium-800k transformer
- **MIDI Processing**: pretty_midi for file manipulation
- **Audio Conversion**: FluidSynth with soundfonts + SciPy for processing
- **Music Theory**: Custom utilities for chord parsing and diatonic scales

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Node.js 18+**
- **FluidSynth** (for high-quality audio):

  ```bash
  # macOS
  brew install fluid-synth

  # Ubuntu/Debian
  sudo apt-get install fluidsynth

  # Windows: Download from https://www.fluidsynth.org/
  ```

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python run.py
```

The backend will be available at `http://localhost:8000`

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:5173`

## 🎯 Usage

### 1. Generate Backing Track

1. Select key signature (C, G, F#, etc.)
2. Set tempo (60-200 BPM)
3. Choose number of bars (4-32)
4. Select chord progression:
   - Use presets (I-V-vi-IV, ii-V-I, etc.)
   - Create custom progression
5. Click "Generate Backing Track"

### 2. Generate Lead Melody

1. **Option A**: Use preset seed melody
   - Click "Preview Seed Melody" to hear the seed
   - Click "Generate Lead Melody" to create full melody
2. **Option B**: Create custom seed melody
   - Add notes with pitch, timing, and duration
   - Generate lead melody with your custom seed

### 3. Listen & Download

- Play individual tracks (backing, lead, combined)
- Visualize MIDI in piano roll view
- Download MIDI files for further editing

## 🔧 Technical Details

### AI Generation Process

1. **Backing Track**: Programmatic generation using music theory
2. **Seed Melody**: Algorithmic or user-defined melodic direction
3. **AMT Generation**:
   - Convert MIDI to anticipation events
   - Feed backing + seed events to AMT model
   - Generate new melody events
   - Convert back to MIDI with guitar range filtering
4. **Fallback**: Algorithmic melody if AI generation fails

### Audio Pipeline

1. **MIDI Creation**: pretty_midi generates MIDI files
2. **FluidSynth**: Converts MIDI to audio using soundfonts
3. **SciPy Processing**: Normalizes and adjusts tempo
4. **Final Output**: High-quality WAV files

### Key Libraries

- **AI/ML**: PyTorch, Transformers, Anticipation Library
- **Audio**: pretty_midi, FluidSynth, SciPy, NumPy
- **Web**: FastAPI, React, TypeScript, Tailwind CSS
- **Music Theory**: Custom chord parsing and diatonic scale utilities

## 📁 Project Structure

```
├── backend/
│   ├── app/
│   │   ├── core/
│   │   │   ├── generator_clean.py    # Main AMT generator
│   │   │   ├── audio_utils.py        # Audio conversion
│   │   │   └── music_utils.py        # Music theory utilities
│   │   └── routes/
│   │       └── generator.py          # API endpoints
│   ├── static/                       # Generated files
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── BackingTrack.tsx      # Backing track UI
│   │   │   ├── LeadMelody.tsx        # Lead melody UI
│   │   │   ├── AudioPlayer.tsx       # Audio playback
│   │   │   └── MidiVisualizer.tsx    # MIDI visualization
│   │   └── App.tsx                   # Main app component
│   └── package.json
└── README.md
```

## 🎼 Example Workflow

1. **Create Backing Track**: C major, 120 BPM, 8 bars, C-Am-F-G progression
2. **Preview Seed**: Listen to the algorithmically generated seed melody
3. **Generate Lead**: AI creates a full 8-bar melody that fits the backing
4. **Combine**: Get a complete song with backing track + AI lead melody
5. **Practice**: Use the generated material for guitar practice

## 🔄 API Endpoints

- `POST /api/backing/chords` - Generate backing track with custom chords
- `POST /api/seed` - Generate preset seed melody
- `POST /api/lead/{generation_id}` - Generate lead melody with preset seed
- `POST /api/lead/{generation_id}/custom` - Generate lead melody with custom seed
- `GET /static/{generation_id}/{filename}` - Download generated files

## 🛠️ Development

### Backend Development

```bash
cd backend
python run.py  # Development server with auto-reload
```

### Frontend Development

```bash
cd frontend
npm run dev    # Development server
npm run build  # Production build
```

### Testing

- Backend: Check API endpoints with curl or Postman
- Frontend: Test UI components and audio playback
- Integration: Verify full workflow from backing track to combined song

## 🎯 Use Cases

- **Guitar Practice**: Generate backing tracks for improvisation practice
- **Songwriting**: Use AI-generated melodies as starting points
- **Learning**: Study how melodies fit with chord progressions
- **Performance**: Create backing tracks for live performances

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly (especially audio generation)
5. Submit a pull request

## 📄 License

This project is open source. Please check individual component licenses.

---

**LeadLine** - Making AI-generated music accessible to guitarists everywhere! 🎸
