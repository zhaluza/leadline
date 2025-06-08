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
    <div className="p-4 bg-gray-800/50 rounded-lg border border-gray-700">
      <h2 className="text-xl font-semibold mb-4 text-blue-300">
        Backing Track Generator
      </h2>

      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              Key
            </label>
            <select
              value={key}
              onChange={(e) => setKey(e.target.value)}
              className="w-full rounded-md border-gray-600 bg-gray-700 text-white shadow-sm focus:border-blue-500 focus:ring-blue-500"
            >
              {keys.map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              Tempo (BPM)
            </label>
            <input
              type="number"
              min="60"
              max="200"
              value={tempo}
              onChange={(e) => setTempo(Number(e.target.value))}
              className="w-full rounded-md border-gray-600 bg-gray-700 text-white shadow-sm focus:border-blue-500 focus:ring-blue-500"
            />
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">
            Number of Bars
          </label>
          <input
            type="number"
            min="4"
            max="32"
            value={numBars}
            onChange={(e) => setNumBars(Number(e.target.value))}
            className="w-full rounded-md border-gray-600 bg-gray-700 text-white shadow-sm focus:border-blue-500 focus:ring-blue-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Chord Progression
          </label>

          <div className="mb-3">
            <select
              onChange={(e) => handleProgressionSelect(e.target.value)}
              className="w-full rounded-md border-gray-600 bg-gray-700 text-white shadow-sm focus:border-blue-500 focus:ring-blue-500"
            >
              {Object.keys(commonProgressions).map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </div>

          <div className="flex gap-2 mb-3">
            <input
              type="text"
              placeholder="Add chord (e.g., C, Am, F#m7)"
              value={customChord}
              onChange={(e) => setCustomChord(e.target.value)}
              className="flex-1 rounded-md border-gray-600 bg-gray-700 text-white shadow-sm focus:border-blue-500 focus:ring-blue-500"
            />
            <button
              onClick={handleAddChord}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors"
            >
              Add
            </button>
          </div>

          <div className="flex flex-wrap gap-2">
            {chordProgression.map((chord, index) => (
              <div
                key={index}
                className="flex items-center gap-1 bg-blue-600 text-white px-3 py-1 rounded-full"
              >
                <span>{chord}</span>
                <button
                  onClick={() => handleRemoveChord(index)}
                  className="ml-1 text-blue-200 hover:text-white"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </div>

        <button
          onClick={handleGenerate}
          disabled={isGenerating || chordProgression.length === 0}
          className={`w-full py-3 px-4 rounded-lg font-semibold text-lg transition-colors
            ${
              isGenerating || chordProgression.length === 0
                ? "bg-gray-700 text-gray-400 cursor-not-allowed"
                : "bg-blue-600 hover:bg-blue-700 text-white shadow-lg hover:shadow-xl"
            }`}
        >
          {isGenerating ? "Generating..." : "Generate Backing Track"}
        </button>
      </div>
    </div>
  );
}
