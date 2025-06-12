import { useState } from "react";
import BackingTrack from "./components/BackingTrack";
import LeadMelody from "./components/LeadMelody";
import AudioPlayer from "./components/AudioPlayer";
import MidiVisualizer from "./components/MidiVisualizer";
import SheetMusicVisualizer from "./components/SheetMusicVisualizer";

function App() {
  const [backingTrackUrl, setBackingTrackUrl] = useState<string | null>(null);
  const [leadMelodyUrl, setLeadMelodyUrl] = useState<string | null>(null);
  const [combinedTrackUrl, setCombinedTrackUrl] = useState<string | null>(null);
  const [combinedMidiUrl, setCombinedMidiUrl] = useState<string | null>(null);
  const [generationId, setGenerationId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showSheetMusic, setShowSheetMusic] = useState(false);

  const handleBackingTrackComplete = (
    generationId: string,
    audioUrl: string
  ) => {
    setBackingTrackUrl(audioUrl);
    setGenerationId(generationId);
    setLeadMelodyUrl(null);
    setCombinedTrackUrl(null);
    setCombinedMidiUrl(null);
    setError(null);
  };

  const handleLeadMelodyComplete = (
    audioUrl: string,
    combinedAudioUrl?: string,
    combinedMidiUrl?: string
  ) => {
    setLeadMelodyUrl(audioUrl);
    if (combinedAudioUrl) setCombinedTrackUrl(combinedAudioUrl);
    if (combinedMidiUrl) setCombinedMidiUrl(combinedMidiUrl);
    setError(null);
  };

  const handleError = (error: string) => {
    setError(error);
    console.error("Generation error:", error);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
      {/* Header */}
      <header className="relative overflow-hidden bg-gradient-to-r from-blue-900/20 to-purple-900/20 border-b border-gray-700/50">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/10 to-purple-600/10"></div>
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center">
            <h1 className="text-4xl md:text-6xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-blue-400 bg-clip-text text-transparent mb-2 animate-gradient">
              LeadLine
            </h1>
            <p className="text-lg text-gray-300">
              AI backing tracks & melodies
            </p>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Error Display */}
        {error && (
          <div className="mb-8 p-6 bg-red-900/30 border border-red-500/50 rounded-xl backdrop-blur-sm">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <svg
                  className="h-5 w-5 text-red-400"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                >
                  <path
                    fillRule="evenodd"
                    d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                    clipRule="evenodd"
                  />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-red-200 font-medium">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Main Workflow */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
          {/* Step 1: Backing Track */}
          <div className="bg-gray-800/40 backdrop-blur-sm p-8 rounded-2xl border border-gray-700/50 shadow-2xl hover:shadow-blue-500/5 transition-all duration-300">
            <div className="flex items-center mb-6">
              <div className="flex-shrink-0 w-10 h-10 bg-blue-600 rounded-full flex items-center justify-center mr-4">
                <span className="text-white font-bold text-lg">1</span>
              </div>
              <div>
                <h2 className="text-2xl font-bold text-blue-300">
                  Backing Track
                </h2>
                <p className="text-gray-400 text-sm">
                  Generate drums, bass, and piano
                </p>
              </div>
            </div>

            <BackingTrack
              onGenerationComplete={handleBackingTrackComplete}
              onError={handleError}
            />

            {backingTrackUrl && (
              <div className="mt-6 p-4 bg-gray-700/30 rounded-xl border border-gray-600/50">
                <AudioPlayer audioUrl={backingTrackUrl} label="Backing Track" />
              </div>
            )}
          </div>

          {/* Step 2: Lead Melody */}
          <div className="bg-gray-800/40 backdrop-blur-sm p-8 rounded-2xl border border-gray-700/50 shadow-2xl hover:shadow-purple-500/5 transition-all duration-300">
            <div className="flex items-center mb-6">
              <div className="flex-shrink-0 w-10 h-10 bg-purple-600 rounded-full flex items-center justify-center mr-4">
                <span className="text-white font-bold text-lg">2</span>
              </div>
              <div>
                <h2 className="text-2xl font-bold text-purple-300">
                  Lead Melody
                </h2>
                <p className="text-gray-400 text-sm">AI-generated melodies</p>
              </div>
            </div>

            <LeadMelody
              generationId={generationId}
              onGenerationComplete={handleLeadMelodyComplete}
              onError={handleError}
            />

            {leadMelodyUrl && (
              <div className="mt-6 p-4 bg-gray-700/30 rounded-xl border border-gray-600/50">
                <AudioPlayer audioUrl={leadMelodyUrl} label="Lead Melody" />
              </div>
            )}

            {combinedTrackUrl && (
              <div className="mt-4 p-4 bg-green-700/20 rounded-xl border border-green-600/50">
                <AudioPlayer
                  audioUrl={combinedTrackUrl}
                  label="Complete Song"
                />
              </div>
            )}
          </div>
        </div>

        {/* MIDI Visualization */}
        {(backingTrackUrl || combinedTrackUrl) && (
          <div className="bg-gray-800/40 backdrop-blur-sm p-8 rounded-2xl border border-gray-700/50 shadow-2xl">
            <div className="flex items-center justify-between mb-8">
              <div className="flex items-center">
                <div className="flex-shrink-0 w-10 h-10 bg-green-600 rounded-full flex items-center justify-center mr-4">
                  <svg
                    className="h-5 w-5 text-white"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"
                    />
                  </svg>
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-green-300">
                    MIDI Visualization
                  </h2>
                  <p className="text-gray-400 text-sm">Visualize your music</p>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <span className="text-gray-400 text-sm font-medium">View:</span>
                <div className="flex bg-gray-700/50 rounded-lg p-1">
                  <button
                    onClick={() => setShowSheetMusic(false)}
                    className={`px-4 py-2 rounded-md text-sm font-medium transition-all duration-200
                      ${
                        !showSheetMusic
                          ? "bg-blue-600 text-white shadow-lg"
                          : "text-gray-300 hover:text-white hover:bg-gray-600/50"
                      }`}
                  >
                    Piano Roll
                  </button>
                  <button
                    disabled={true}
                    onClick={() => setShowSheetMusic(true)}
                    className={`px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 opacity-50 cursor-not-allowed
                      ${
                        showSheetMusic
                          ? "bg-blue-600 text-white shadow-lg"
                          : "text-gray-300"
                      }`}
                  >
                    Sheet Music
                  </button>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {backingTrackUrl && (
                <div className="bg-gray-700/30 rounded-xl p-6 border border-gray-600/50">
                  {showSheetMusic ? (
                    <SheetMusicVisualizer
                      midiUrl={`http://localhost:8000/static/${generationId}/backing.mid`}
                      label="Backing Track Sheet Music"
                      height={250}
                      width={400}
                    />
                  ) : (
                    <MidiVisualizer
                      midiUrl={`http://localhost:8000/static/${generationId}/backing.mid`}
                      label="Backing Track Piano Roll"
                      height={200}
                      width={400}
                    />
                  )}
                </div>
              )}

              {combinedMidiUrl && (
                <div className="bg-gray-700/30 rounded-xl p-6 border border-gray-600/50">
                  {showSheetMusic ? (
                    <SheetMusicVisualizer
                      midiUrl={combinedMidiUrl}
                      label="Complete Song Sheet Music"
                      height={250}
                      width={400}
                    />
                  ) : (
                    <MidiVisualizer
                      midiUrl={combinedMidiUrl}
                      label="Complete Song Piano Roll"
                      height={200}
                      width={400}
                    />
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
