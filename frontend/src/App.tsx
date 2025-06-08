import { useState } from "react";
import BackingTrack from "./components/BackingTrack";
import LeadMelody from "./components/LeadMelody";
import AudioPlayer from "./components/AudioPlayer";

function App() {
  const [backingTrackUrl, setBackingTrackUrl] = useState<string | null>(null);
  const [leadMelodyUrl, setLeadMelodyUrl] = useState<string | null>(null);
  const [combinedTrackUrl, setCombinedTrackUrl] = useState<string | null>(null);
  const [generationId, setGenerationId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleBackingTrackComplete = (
    generationId: string,
    audioUrl: string
  ) => {
    setBackingTrackUrl(audioUrl);
    setGenerationId(generationId);
    setLeadMelodyUrl(null);
    setCombinedTrackUrl(null);
    setError(null);
  };

  const handleLeadMelodyComplete = (
    audioUrl: string,
    combinedAudioUrl?: string
  ) => {
    setLeadMelodyUrl(audioUrl);
    if (combinedAudioUrl) setCombinedTrackUrl(combinedAudioUrl);
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
            Melody Machine
          </h1>
          <p className="text-center text-gray-300 mt-2 mb-4">
            AI-powered songwriting tool for guitarists
          </p>
          <div className="text-center text-gray-400 text-sm max-w-2xl mx-auto">
            <p>
              Create custom backing tracks with your chosen chord progressions,
              then generate AI melodies to complete your songs.
            </p>
          </div>
        </header>

        {error && (
          <div className="mb-6 p-4 bg-red-900/50 border border-red-500 rounded-lg text-red-200">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="bg-gray-800/50 p-6 rounded-lg border border-gray-700">
            <h2 className="text-xl font-semibold mb-4 text-blue-300">
              Step 1: Create Backing Track
            </h2>
            <p className="text-gray-400 text-sm mb-4">
              Choose your key, tempo, and chord progression. The backing track
              includes drums, bass, and piano.
            </p>
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
              Step 2: Generate Melody
            </h2>
            <p className="text-gray-400 text-sm mb-4">
              Once you have a backing track, generate an AI-powered lead melody
              that fits the chord progression.
            </p>
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
            {combinedTrackUrl && (
              <div className="mt-4">
                <AudioPlayer
                  audioUrl={combinedTrackUrl}
                  label="Complete Song"
                />
              </div>
            )}
          </div>
        </div>

        {backingTrackUrl && (
          <div className="mt-8 text-center">
            <p className="text-gray-400 text-sm">
              💡 Tip: Use the backing track for practice, or combine it with the
              melody for a complete song!
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
