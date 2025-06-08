import { useState } from "react";
import { BackingTrack } from "./components/BackingTrack";
import { LeadMelody } from "./components/LeadMelody";
import { AudioPlayer } from "./components/AudioPlayer";
import "./App.css";

function App() {
  const [backingTrackId, setBackingTrackId] = useState<string>();
  const [backingAudioUrl, setBackingAudioUrl] = useState<string>();
  const [leadAudioUrl, setLeadAudioUrl] = useState<string>();
  const [combinedAudioUrl, setCombinedAudioUrl] = useState<string>();

  const handleBackingTrackComplete = (
    generationId: string,
    audioUrl: string
  ) => {
    setBackingTrackId(generationId);
    setBackingAudioUrl(audioUrl);
    // Reset lead-related state when new backing track is generated
    setLeadAudioUrl(undefined);
    setCombinedAudioUrl(undefined);
  };

  const handleLeadMelodyComplete = (
    generationId: string,
    audioUrl: string,
    midiUrl: string,
    combinedAudioUrl: string
  ) => {
    setLeadAudioUrl(audioUrl);
    setCombinedAudioUrl(combinedAudioUrl);
  };

  return (
    <div className="min-h-screen bg-gray-100 py-6 flex flex-col justify-center sm:py-12">
      <div className="relative py-3 sm:max-w-xl sm:mx-auto">
        <div className="relative px-4 py-10 bg-white mx-8 md:mx-0 shadow rounded-3xl sm:p-10">
          <div className="max-w-md mx-auto">
            <div className="divide-y divide-gray-200">
              <div className="py-8 text-base leading-6 space-y-4 text-gray-700 sm:text-lg sm:leading-7">
                <h1 className="text-3xl font-bold text-center mb-8 text-indigo-600">
                  AMT Backing Track Generator
                </h1>

                <div className="space-y-8">
                  {/* Backing Track Section */}
                  <div>
                    <BackingTrack
                      onGenerationComplete={handleBackingTrackComplete}
                    />
                    {backingAudioUrl && (
                      <div className="mt-4">
                        <AudioPlayer
                          audioUrl={`http://localhost:8000${backingAudioUrl}`}
                          title="Backing Track"
                        />
                      </div>
                    )}
                  </div>

                  {/* Lead Melody Section */}
                  <div>
                    <LeadMelody
                      onGenerationComplete={handleLeadMelodyComplete}
                      backingTrackId={backingTrackId}
                    />
                    {leadAudioUrl && (
                      <div className="mt-4 space-y-4">
                        <AudioPlayer
                          audioUrl={`http://localhost:8000${leadAudioUrl}`}
                          title="Lead Melody"
                        />
                        {combinedAudioUrl && (
                          <AudioPlayer
                            audioUrl={`http://localhost:8000${combinedAudioUrl}`}
                            title="Combined Track"
                          />
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
