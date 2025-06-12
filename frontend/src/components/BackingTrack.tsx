import { useState } from "react";

interface BackingTrackProps {
  onGenerationComplete: (generationId: string, audioUrl: string) => void;
  onError: (error: string) => void;
}

export default function BackingTrack({
  onGenerationComplete,
  onError,
}: BackingTrackProps) {
  const [isGenerating, setIsGenerating] = useState(false);
  const [numBars, setNumBars] = useState(8);
  const [tempo, setTempo] = useState(120);
  const [key, setKey] = useState("C");
  const [chordProgression, setChordProgression] = useState<string[]>([
    "C",
    "Am",
    "F",
    "G",
  ]);
  const [customChord, setCustomChord] = useState("");

  const keys = [
    "C",
    "C#",
    "Db",
    "D",
    "D#",
    "Eb",
    "E",
    "F",
    "F#",
    "Gb",
    "G",
    "G#",
    "Ab",
    "A",
    "A#",
    "Bb",
    "B",
  ];

  const commonProgressions = {
    "I-V-vi-IV (Pop)": ["C", "G", "Am", "F"],
    "ii-V-I (Jazz)": ["Dm", "G", "C"],
    "I-vi-ii-V (Jazz)": ["C", "Am", "Dm", "G"],
    "vi-IV-I-V (Pop)": ["Am", "F", "C", "G"],
    "I-IV-V (Blues)": ["C", "F", "G"],
    Custom: [],
  };

  const handleAddChord = () => {
    if (customChord.trim()) {
      setChordProgression([...chordProgression, customChord.trim()]);
      setCustomChord("");
    }
  };

  const handleRemoveChord = (index: number) => {
    setChordProgression(chordProgression.filter((_, i) => i !== index));
  };

  const handleProgressionSelect = (progressionName: string) => {
    if (progressionName === "Custom") {
      setChordProgression([]);
    } else {
      setChordProgression(
        commonProgressions[progressionName as keyof typeof commonProgressions]
      );
    }
  };

  const handleGenerate = async () => {
    if (chordProgression.length === 0) {
      onError("Please add at least one chord to the progression");
      return;
    }

    setIsGenerating(true);
    try {
      const response = await fetch("http://localhost:8000/api/backing/chords", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          num_bars: numBars,
          tempo: tempo,
          key: key,
          chord_progression: chordProgression,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to generate backing track");
      }

      const data = await response.json();
      onGenerationComplete(
        data.generation_id,
        `http://localhost:8000${data.audio_url}`
      );
    } catch (error) {
      onError(
        error instanceof Error
          ? error.message
          : "Failed to generate backing track"
      );
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Basic Settings */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="space-y-2">
          <label className="block text-sm font-semibold text-gray-300">
            Key
          </label>
          <select
            value={key}
            onChange={(e) => setKey(e.target.value)}
            className="w-full px-4 py-3 rounded-xl border border-gray-600 bg-gray-700/50 text-white shadow-sm focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all duration-200 hover:bg-gray-700/70"
          >
            {keys.map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-semibold text-gray-300">
            Tempo (BPM)
          </label>
          <input
            type="number"
            min="60"
            max="200"
            value={tempo}
            onChange={(e) => setTempo(Number(e.target.value))}
            className="w-full px-4 py-3 rounded-xl border border-gray-600 bg-gray-700/50 text-white shadow-sm focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all duration-200 hover:bg-gray-700/70"
          />
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-semibold text-gray-300">
            Number of Bars
          </label>
          <input
            type="number"
            min="4"
            max="32"
            value={numBars}
            onChange={(e) => setNumBars(Number(e.target.value))}
            className="w-full px-4 py-3 rounded-xl border border-gray-600 bg-gray-700/50 text-white shadow-sm focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all duration-200 hover:bg-gray-700/70"
          />
        </div>
      </div>

      {/* Chord Progression */}
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-semibold text-gray-300 mb-3">
            Chord Progression
          </label>

          {/* Preset Progressions */}
          <div className="mb-4">
            <label className="block text-xs text-gray-400 mb-2">
              Quick Start:
            </label>
            <select
              onChange={(e) => handleProgressionSelect(e.target.value)}
              className="w-full px-4 py-3 rounded-xl border border-gray-600 bg-gray-700/50 text-white shadow-sm focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all duration-200 hover:bg-gray-700/70"
            >
              {Object.keys(commonProgressions).map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </div>

          {/* Custom Chord Input */}
          <div className="flex gap-3 mb-4">
            <input
              type="text"
              placeholder="Add chord (e.g., C, Am, F#m7)"
              value={customChord}
              onChange={(e) => setCustomChord(e.target.value)}
              onKeyPress={(e) => e.key === "Enter" && handleAddChord()}
              className="flex-1 px-4 py-3 rounded-xl border border-gray-600 bg-gray-700/50 text-white shadow-sm focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all duration-200 hover:bg-gray-700/70 placeholder-gray-400"
            />
            <button
              onClick={handleAddChord}
              disabled={!customChord.trim()}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:text-gray-400 text-white rounded-xl transition-all duration-200 font-medium shadow-lg hover:shadow-xl disabled:shadow-none"
            >
              Add
            </button>
          </div>

          {/* Chord Display */}
          <div className="min-h-[60px] p-4 bg-gray-700/30 rounded-xl border border-gray-600/50">
            {chordProgression.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {chordProgression.map((chord, index) => (
                  <div
                    key={index}
                    className="flex items-center gap-2 bg-gradient-to-r from-blue-600 to-blue-700 text-white px-4 py-2 rounded-full shadow-lg hover:shadow-xl transition-all duration-200 group"
                  >
                    <span className="font-semibold">{chord}</span>
                    <button
                      onClick={() => handleRemoveChord(index)}
                      className="ml-1 text-blue-200 hover:text-white transition-colors duration-200 opacity-70 hover:opacity-100"
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
            ) : (
              <div className="text-gray-400 text-center py-2">
                No chords added yet. Select a preset or add custom chords above.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Generate Button */}
      <button
        onClick={handleGenerate}
        disabled={isGenerating || chordProgression.length === 0}
        className={`w-full py-4 px-6 rounded-xl font-bold text-lg transition-all duration-300 shadow-lg
          ${
            isGenerating || chordProgression.length === 0
              ? "bg-gray-600 text-gray-400 cursor-not-allowed shadow-none"
              : "bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white shadow-blue-500/25 hover:shadow-blue-500/40 hover:scale-[1.02] active:scale-[0.98]"
          }`}
      >
        {isGenerating ? (
          <div className="flex items-center justify-center gap-3">
            <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
            <span>Generating Backing Track...</span>
          </div>
        ) : (
          "Generate Backing Track"
        )}
      </button>
    </div>
  );
}
