import { useState } from "react";

interface BackingTrackProps {
  onGenerationComplete: (generationId: string, audioUrl: string) => void;
}

export const BackingTrack = ({ onGenerationComplete }: BackingTrackProps) => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [numBars, setNumBars] = useState(8);
  const [tempo, setTempo] = useState(120);

  const generateBackingTrack = async () => {
    setIsLoading(true);
    setError(null);

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
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      onGenerationComplete(data.generation_id, data.audio_url);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setIsLoading(false);
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
          onClick={generateBackingTrack}
          disabled={isLoading}
          className={`w-full py-2 px-4 rounded-md text-white font-medium ${
            isLoading
              ? "bg-indigo-400 cursor-not-allowed"
              : "bg-indigo-600 hover:bg-indigo-700"
          }`}
        >
          {isLoading ? "Generating..." : "Generate Backing Track"}
        </button>

        {error && <div className="text-red-600 text-sm mt-2">{error}</div>}
      </div>
    </div>
  );
};
