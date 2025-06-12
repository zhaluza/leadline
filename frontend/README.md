# LeadLine Frontend

A React-based frontend for the LeadLine AI-powered backing tracks and lead melody generator.

## Features

- **Backing Track Generation**: Create custom backing tracks with drums, bass, and piano
- **AI Lead Melody Generation**: Generate AI-powered lead melodies that fit your chord progressions
- **MIDI Visualization**:
  - **Piano Roll View**: Interactive piano roll visualization with play/pause functionality
  - **Sheet Music View**: Traditional music notation display using VexFlow
  - Real-time playback with visual feedback
  - Progress indicators and note highlighting
- **Audio Playback**: Listen to generated tracks with built-in audio players
- **Custom Seed Melodies**: Create custom seed notes for more personalized melodies

## Technologies Used

- **React 19** with TypeScript
- **Vite** for fast development and building
- **Tailwind CSS** for styling
- **Tone.js** for audio processing and MIDI playback
- **@tonejs/midi** for MIDI file parsing
- **VexFlow** for sheet music rendering
- **Canvas API** for custom piano roll visualization

## Development

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

```bash
npm install
```

### Running the Development Server

```bash
npm run dev
```

The app will be available at `http://localhost:5173`

### Building for Production

```bash
npm run build
```

## MIDI Visualization Features

### Piano Roll View

- Interactive canvas-based visualization
- Color-coded notes based on velocity
- Real-time playback with visual playhead
- Note highlighting during playback
- Time and pitch axis labels
- Progress bar with current time display

### Sheet Music View

- Traditional music notation using VexFlow
- Automatic conversion from MIDI to sheet music
- Treble clef with 4/4 time signature
- Limited to first 16 notes for demo purposes

### Controls

- Play/Pause button for MIDI playback
- Toggle between piano roll and sheet music views
- Responsive design for different screen sizes

## Demo Features

For the demo, the app includes:

- Default settings to avoid known bugs (tempo, key, chord modifications)
- Impressive MIDI visualizations for both backing tracks and combined songs
- Interactive playback controls
- Professional-looking UI with dark theme
- Real-time visual feedback during playback

## API Integration

The frontend communicates with the LeadLine backend API running on `http://localhost:8000`:

- Backing track generation
- Lead melody generation with custom seeds
- MIDI and audio file serving
- Combined track creation

## Project Structure

```
src/
├── components/          # React components
│   ├── BackingTrack.tsx # Backing track generation UI
│   └── LeadMelody.tsx   # Lead melody generation UI
├── main.tsx            # Application entry point
└── App.tsx             # Main application component
```

## Contributing

1. Follow the existing code style and TypeScript patterns
2. Ensure all new features include proper error handling
3. Test audio generation features thoroughly
4. Update documentation for any API changes
