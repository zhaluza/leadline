import { useState } from "react";

interface LeadMelodyProps {
  generationId: string | null;
  onGenerationComplete: (audioUrl: string, combinedAudioUrl?: string) => void;
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
    <div className="p-4 bg-gray-800/50 rounded-lg border border-gray-700">
      <h2 className="text-xl font-semibold mb-4 text-blue-300">
        Lead Melody Generator
      </h2>

      <div className="space-y-4">
        {/* Seed Melody Preview Section */}
        <div className="p-3 bg-gray-700/30 rounded border border-gray-600">
          <h3 className="text-sm font-medium text-gray-300 mb-2">
            Seed Melody Preview
          </h3>
          <p className="text-xs text-gray-400 mb-3">
            Preview the 1-bar seed melody that will be used to generate the full
            8-bar lead melody
          </p>

          <div className="flex gap-2">
            <button
              onClick={handlePreviewSeed}
              disabled={isPreviewing || !generationId}
              className={`px-4 py-2 rounded text-sm font-medium transition-colors
                ${
                  !generationId
                    ? "bg-gray-600 text-gray-400 cursor-not-allowed"
                    : isPreviewing
                    ? "bg-gray-600 text-gray-400 cursor-not-allowed"
                    : "bg-green-600 hover:bg-green-700 text-white"
                }`}
            >
              {isPreviewing ? "Generating Preview..." : "Preview Seed Melody"}
            </button>

            {seedPreviewUrl && (
              <audio controls className="flex-1 h-10" src={seedPreviewUrl}>
                Your browser does not support the audio element.
              </audio>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="useCustomSeed"
            checked={useCustomSeed}
            onChange={(e) => setUseCustomSeed(e.target.checked)}
            className="rounded border-gray-600 bg-gray-700 text-blue-500 focus:ring-blue-500"
          />
          <label htmlFor="useCustomSeed" className="text-gray-300 text-sm">
            Use custom seed melody (add notes for first 4 measures)
          </label>
        </div>

        {useCustomSeed && (
          <div className="space-y-3 p-3 bg-gray-700/50 rounded border border-gray-600">
            <h3 className="text-sm font-medium text-gray-300">
              Seed Melody Notes (First 4 Measures)
            </h3>

            <div className="grid grid-cols-4 gap-2 text-xs">
              <div>
                <label className="block text-gray-400 mb-1">Note</label>
                <select
                  value={newNotePitch}
                  onChange={(e) => setNewNotePitch(Number(e.target.value))}
                  className="w-full rounded border-gray-600 bg-gray-700 text-white"
                >
                  {pitches.map((pitch) => (
                    <option key={pitch} value={pitch}>
                      {getNoteName(pitch)}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-gray-400 mb-1">
                  Start (beats)
                </label>
                <input
                  type="number"
                  min="0"
                  max="16"
                  step="0.25"
                  value={newNoteStart}
                  onChange={(e) => setNewNoteStart(Number(e.target.value))}
                  className="w-full rounded border-gray-600 bg-gray-700 text-white"
                />
              </div>

              <div>
                <label className="block text-gray-400 mb-1">
                  Duration (beats)
                </label>
                <input
                  type="number"
                  min="0.25"
                  max="16"
                  step="0.25"
                  value={newNoteDuration}
                  onChange={(e) => setNewNoteDuration(Number(e.target.value))}
                  className="w-full rounded border-gray-600 bg-gray-700 text-white"
                />
              </div>

              <div className="flex items-end">
                <button
                  onClick={addSeedNote}
                  className="w-full px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors"
                >
                  Add
                </button>
              </div>
            </div>

            {seedNotes.length > 0 && (
              <div className="space-y-1">
                <div className="text-xs text-gray-400">Added notes:</div>
                {seedNotes.map((note, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between bg-gray-600 px-2 py-1 rounded text-xs"
                  >
                    <span className="text-white">
                      {getNoteName(note.pitch)} at {note.start}s for{" "}
                      {note.duration}s
                    </span>
                    <button
                      onClick={() => removeSeedNote(index)}
                      className="text-red-400 hover:text-red-300"
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        <button
          onClick={handleGenerate}
          disabled={isGenerating || !generationId}
          className={`w-full py-3 px-4 rounded-lg font-semibold text-lg transition-colors
            ${
              !generationId
                ? "bg-gray-700 text-gray-400 cursor-not-allowed"
                : isGenerating
                ? "bg-gray-700 text-gray-400 cursor-not-allowed"
                : "bg-blue-600 hover:bg-blue-700 text-white shadow-lg hover:shadow-xl"
            }`}
        >
          {!generationId
            ? "Generate Backing Track First"
            : isGenerating
            ? "Generating..."
            : "Generate Lead Melody (4-measure seed + AI completion)"}
        </button>

        {showNoBackingTrackMsg && (
          <div className="text-yellow-400 text-sm mt-2">
            Please generate a backing track first
          </div>
        )}
      </div>
    </div>
  );
}
