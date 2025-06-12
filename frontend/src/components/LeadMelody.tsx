import { useState } from "react";

interface LeadMelodyProps {
  generationId: string | null;
  onGenerationComplete: (
    audioUrl: string,
    combinedAudioUrl?: string,
    combinedMidiUrl?: string
  ) => void;
  onError: (error: string) => void;
}

interface SeedNote {
  pitch: number;
  start: number;
  duration: number;
}

export default function LeadMelody({
  generationId,
  onGenerationComplete,
  onError,
}: LeadMelodyProps) {
  const [isGenerating, setIsGenerating] = useState(false);
  const [isPreviewing, setIsPreviewing] = useState(false);
  const [showNoBackingTrackMsg, setShowNoBackingTrackMsg] = useState(false);
  const [seedNotes, setSeedNotes] = useState<SeedNote[]>([]);
  const [useCustomSeed, setUseCustomSeed] = useState(false);
  const [newNotePitch, setNewNotePitch] = useState(72);
  const [newNoteStart, setNewNoteStart] = useState(0);
  const [newNoteDuration, setNewNoteDuration] = useState(1);
  const [seedPreviewUrl, setSeedPreviewUrl] = useState<string | null>(null);

  const noteNames = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
  ];
  const pitches = Array.from({ length: 25 }, (_, i) => i + 60); // C4 to C6

  const getNoteName = (pitch: number) => {
    return noteNames[pitch % 12] + Math.floor(pitch / 12);
  };

  const addSeedNote = () => {
    const newNote: SeedNote = {
      pitch: newNotePitch,
      start: newNoteStart,
      duration: newNoteDuration,
    };
    setSeedNotes([...seedNotes, newNote]);
  };

  const removeSeedNote = (index: number) => {
    setSeedNotes(seedNotes.filter((_, i) => i !== index));
  };

  const handlePreviewSeed = async () => {
    if (!generationId) {
      setShowNoBackingTrackMsg(true);
      onError("Please generate a backing track first");
      return;
    }
    setShowNoBackingTrackMsg(false);
    setIsPreviewing(true);

    try {
      const response = await fetch(
        `http://localhost:8000/api/lead/${generationId}/preview-seed`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            num_bars: 8,
            tempo: 120,
            key: 60, // C major
          }),
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to preview seed melody");
      }

      const data = await response.json();
      setSeedPreviewUrl(`http://localhost:8000${data.audio_url}`);
    } catch (error) {
      onError(
        error instanceof Error ? error.message : "Failed to preview seed melody"
      );
    } finally {
      setIsPreviewing(false);
    }
  };

  const handleGenerate = async () => {
    if (!generationId) {
      setShowNoBackingTrackMsg(true);
      onError("Please generate a backing track first");
      return;
    }
    setShowNoBackingTrackMsg(false);
    setIsGenerating(true);

    try {
      const endpoint =
        useCustomSeed && seedNotes.length > 0
          ? `/api/lead/${generationId}/custom`
          : `/api/lead/${generationId}`;

      const requestBody =
        useCustomSeed && seedNotes.length > 0
          ? {
              num_bars: 8,
              tempo: 120,
              key: "C",
              seed_notes: seedNotes,
            }
          : {
              num_bars: 8,
              tempo: 120,
            };

      const response = await fetch(`http://localhost:8000${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to generate lead melody");
      }

      const data = await response.json();
      onGenerationComplete(
        `http://localhost:8000${data.audio_url}`,
        data.combined_audio_url
          ? `http://localhost:8000${data.combined_audio_url}`
          : undefined,
        data.combined_midi_url
          ? `http://localhost:8000${data.combined_midi_url}`
          : undefined
      );
    } catch (error) {
      onError(
        error instanceof Error
          ? error.message
          : "Failed to generate lead melody"
      );
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Seed Melody Preview Section */}
      <div className="p-6 bg-gradient-to-r from-purple-900/20 to-blue-900/20 rounded-xl border border-purple-500/30 backdrop-blur-sm">
        <div className="flex items-center mb-4">
          <div className="flex-shrink-0 w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center mr-3">
            <svg
              className="h-4 w-4 text-white"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M14.828 14.828a4 4 0 01-5.656 0M9 10h1m4 0h1m-6 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </div>
          <div>
            <h3 className="text-lg font-semibold text-purple-300">
              Seed Melody Preview
            </h3>
            <p className="text-sm text-gray-400">
              Preview the 1-bar seed melody that will be used to generate the
              full 8-bar lead melody
            </p>
          </div>
        </div>

        <div className="flex gap-3 items-center">
          <button
            onClick={handlePreviewSeed}
            disabled={isPreviewing || !generationId}
            className={`px-6 py-3 rounded-xl text-sm font-medium transition-all duration-200 shadow-lg
              ${
                !generationId
                  ? "bg-gray-600 text-gray-400 cursor-not-allowed shadow-none"
                  : isPreviewing
                  ? "bg-gray-600 text-gray-400 cursor-not-allowed shadow-none"
                  : "bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 text-white shadow-purple-500/25 hover:shadow-purple-500/40 hover:scale-[1.02] active:scale-[0.98]"
              }`}
          >
            {isPreviewing ? (
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                <span>Generating Preview...</span>
              </div>
            ) : (
              "Preview Seed Melody"
            )}
          </button>

          {seedPreviewUrl && (
            <div className="flex-1">
              <audio
                controls
                className="w-full h-12 rounded-xl bg-gray-700/50 border border-gray-600/50"
                src={seedPreviewUrl}
              >
                Your browser does not support the audio element.
              </audio>
            </div>
          )}
        </div>
      </div>

      {/* Custom Seed Toggle */}
      <div className="flex items-center gap-3 p-4 bg-gray-700/30 rounded-xl border border-gray-600/50">
        <input
          type="checkbox"
          id="useCustomSeed"
          checked={useCustomSeed}
          onChange={(e) => setUseCustomSeed(e.target.checked)}
          className="w-5 h-5 rounded border-gray-600 bg-gray-700 text-purple-500 focus:ring-2 focus:ring-purple-500/20 transition-all duration-200"
        />
        <label
          htmlFor="useCustomSeed"
          className="text-gray-300 text-sm font-medium"
        >
          Use custom seed melody (add notes for first measure)
        </label>
      </div>

      {/* Custom Seed Input */}
      {useCustomSeed && (
        <div className="space-y-4 p-6 bg-gray-700/30 rounded-xl border border-gray-600/50">
          <div className="flex items-center mb-4">
            <div className="flex-shrink-0 w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center mr-3">
              <svg
                className="h-3 w-3 text-white"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 6v6m0 0v6m0-6h6m-6 0H6"
                />
              </svg>
            </div>
            <h3 className="text-lg font-semibold text-blue-300">
              Seed Melody Notes (First Measure)
            </h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-400">
                Note
              </label>
              <select
                value={newNotePitch}
                onChange={(e) => setNewNotePitch(Number(e.target.value))}
                className="w-full px-3 py-2 rounded-lg border border-gray-600 bg-gray-700/50 text-white shadow-sm focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all duration-200 hover:bg-gray-700/70"
              >
                {pitches.map((pitch) => (
                  <option key={pitch} value={pitch}>
                    {getNoteName(pitch)}
                  </option>
                ))}
              </select>
            </div>

            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-400">
                Start (beats)
              </label>
              <input
                type="number"
                min="0"
                max="4"
                step="0.25"
                value={newNoteStart}
                onChange={(e) => setNewNoteStart(Number(e.target.value))}
                className="w-full px-3 py-2 rounded-lg border border-gray-600 bg-gray-700/50 text-white shadow-sm focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all duration-200 hover:bg-gray-700/70"
              />
            </div>

            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-400">
                Duration (beats)
              </label>
              <input
                type="number"
                min="0.25"
                max="4"
                step="0.25"
                value={newNoteDuration}
                onChange={(e) => setNewNoteDuration(Number(e.target.value))}
                className="w-full px-3 py-2 rounded-lg border border-gray-600 bg-gray-700/50 text-white shadow-sm focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all duration-200 hover:bg-gray-700/70"
              />
            </div>

            <div className="flex items-end">
              <button
                onClick={addSeedNote}
                className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-all duration-200 font-medium shadow-lg hover:shadow-xl hover:scale-[1.02] active:scale-[0.98]"
              >
                Add Note
              </button>
            </div>
          </div>

          {/* Added Notes Display */}
          {seedNotes.length > 0 && (
            <div className="space-y-3">
              <div className="text-sm font-medium text-gray-400">
                Added notes:
              </div>
              <div className="space-y-2">
                {seedNotes.map((note, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between bg-gray-600/50 px-4 py-3 rounded-lg border border-gray-500/50"
                  >
                    <span className="text-white font-medium">
                      {getNoteName(note.pitch)} at {note.start}s for{" "}
                      {note.duration}s
                    </span>
                    <button
                      onClick={() => removeSeedNote(index)}
                      className="text-red-400 hover:text-red-300 transition-colors duration-200 p-1 rounded hover:bg-red-500/20"
                    >
                      <svg
                        className="w-4 h-4"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M6 18L18 6M6 6l12 12"
                        />
                      </svg>
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Generate Button */}
      <button
        onClick={handleGenerate}
        disabled={isGenerating || !generationId}
        className={`w-full py-4 px-6 rounded-xl font-bold text-lg transition-all duration-300 shadow-lg
          ${
            !generationId
              ? "bg-gray-600 text-gray-400 cursor-not-allowed shadow-none"
              : isGenerating
              ? "bg-gray-600 text-gray-400 cursor-not-allowed shadow-none"
              : "bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 text-white shadow-purple-500/25 hover:shadow-purple-500/40 hover:scale-[1.02] active:scale-[0.98]"
          }`}
      >
        {!generationId ? (
          "Generate Backing Track First"
        ) : isGenerating ? (
          <div className="flex items-center justify-center gap-3">
            <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
            <span>Generating Lead Melody...</span>
          </div>
        ) : (
          "Generate Lead Melody"
        )}
      </button>

      {/* Warning Message */}
      {showNoBackingTrackMsg && (
        <div className="p-4 bg-yellow-900/20 border border-yellow-500/30 rounded-xl backdrop-blur-sm">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <svg
                className="h-5 w-5 text-yellow-400"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z"
                />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-yellow-200 font-medium">
                Please generate a backing track first
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
