# LeadLine Frontend

The frontend application for LeadLine - an AI-powered guitar backing track and lead melody generator.

## Features

- **Backing Track Generation**: Create custom backing tracks with chord progressions
- **Lead Melody Generation**: Generate AI-powered lead melodies over backing tracks
- **Seed Melody Customization**: Customize the seed melody for more personalized results
- **Real-time Audio Preview**: Preview generated tracks and melodies before downloading

## Tech Stack

- **React 19** with TypeScript
- **Vite** for fast development and building
- **Tailwind CSS** for styling
- **ESLint** for code quality

## Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Run linting
npm run lint
```

## API Integration

This frontend connects to the LeadLine backend API running on `http://localhost:8000` for:

- Backing track generation with custom chord progressions
- Lead melody generation using Anticipatory Music Transformers (AMT)
- Audio file serving and streaming

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
