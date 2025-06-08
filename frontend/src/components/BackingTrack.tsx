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

  const handleGenerate = async () => {
    setIsGenerating(true);
    try {
      const response = await fetch("http://localhost:8000/api/backing", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          num_bars: numBars,
          tempo: tempo,
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
    <div className="p-4 bg-white rounded-lg shadow-md">
      <h2 className="text-xl font-bold mb-4">Backing Track Generator</h2>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700">
            Number of Bars
            <input
              type="number"
              min="4"
              max="32"
              value={numBars}
              onChange={(e) => setNumBars(Number(e.target.value))}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
            />
          </label>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">
            Tempo (BPM)
            <input
              type="number"
              min="60"
              max="200"
              value={tempo}
              onChange={(e) => setTempo(Number(e.target.value))}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
            />
          </label>
        </div>

        <button
          onClick={handleGenerate}
          disabled={isGenerating}
          className={`w-full py-2 px-4 rounded-lg font-medium transition-colors
            ${
              isGenerating
                ? "bg-gray-700 text-gray-300 cursor-not-allowed"
                : "bg-indigo-800 hover:bg-indigo-900 text-white shadow-md hover:shadow-lg"
            }`}
        >
          {isGenerating ? "Generating..." : "Generate Backing Track"}
        </button>
      </div>
    </div>
  );
}
