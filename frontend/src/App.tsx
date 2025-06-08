import { useState } from "react";
import BackingTrack from "./components/BackingTrack";
import LeadMelody from "./components/LeadMelody";
import AudioPlayer from "./components/AudioPlayer";
import "./App.css";

function App() {
  const [backingTrackUrl, setBackingTrackUrl] = useState<string | null>(null);
  const [leadMelodyUrl, setLeadMelodyUrl] = useState<string | null>(null);
  const [generationId, setGenerationId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleBackingTrackComplete = (
    generationId: string,
    audioUrl: string
  ) => {
    setBackingTrackUrl(audioUrl);
    setGenerationId(generationId);
    setError(null);
  };

  const handleLeadMelodyComplete = (audioUrl: string) => {
    setLeadMelodyUrl(audioUrl);
    setError(null);
  };

  const handleError = (error: string) => {
    setError(error);
    console.error("Generation error:", error);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
        <header className="mb-8">
          <h1 className="text-3xl md:text-4xl font-bold text-center text-blue-400">
            AMT Backing Track Generator
          </h1>
          <p className="text-center text-gray-300 mt-2">
            Generate AI-powered backing tracks and lead melodies
          </p>
        </header>

        {error && (
          <div className="mb-6 p-4 bg-red-900/50 border border-red-500 rounded-lg text-red-200">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="bg-gray-800/50 p-6 rounded-lg border border-gray-700">
            <h2 className="text-xl font-semibold mb-4 text-blue-300">
              Backing Track
            </h2>
            <BackingTrack
              onGenerationComplete={handleBackingTrackComplete}
              onError={handleError}
            />
            {backingTrackUrl && (
              <div className="mt-4">
                <AudioPlayer audioUrl={backingTrackUrl} label="Backing Track" />
              </div>
            )}
          </div>

          <div className="bg-gray-800/50 p-6 rounded-lg border border-gray-700">
            <h2 className="text-xl font-semibold mb-4 text-blue-300">
              Lead Melody
            </h2>
            <LeadMelody
              generationId={generationId}
              onGenerationComplete={handleLeadMelodyComplete}
              onError={handleError}
            />
            {leadMelodyUrl && (
              <div className="mt-4">
                <AudioPlayer audioUrl={leadMelodyUrl} label="Lead Melody" />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
