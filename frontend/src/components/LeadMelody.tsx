import { useState } from "react";

interface LeadMelodyProps {
  onGenerationComplete: (
    generationId: string,
    audioUrl: string,
    midiUrl: string,
    combinedAudioUrl: string
  ) => void;
  backingTrackId?: string;
}

export const LeadMelody = ({
  onGenerationComplete,
  backingTrackId,
}: LeadMelodyProps) => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [numBars, setNumBars] = useState(8);
  const [tempo, setTempo] = useState(120);
  const [key, setKey] = useState(60); // Middle C

  const generateLeadMelody = async () => {
    if (!backingTrackId) {
      setError("Please generate a backing track first");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch("http://localhost:8000/api/lead", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          num_bars: numBars,
          tempo: tempo,
          key: key,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      onGenerationComplete(
        data.generation_id,
        data.audio_url,
        data.midi_url,
        data.combined_audio_url
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setIsLoading(false);
    }
  };

  const keyOptions = [
    { value: 60, label: "C" },
    { value: 62, label: "D" },
    { value: 64, label: "E" },
    { value: 65, label: "F" },
    { value: 67, label: "G" },
    { value: 69, label: "A" },
    { value: 71, label: "B" },
  ];

  return (
    <div className="p-4 bg-white rounded-lg shadow-md">
      <h2 className="text-xl font-bold mb-4">Lead Melody Generator</h2>

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

        <div>
          <label className="block text-sm font-medium text-gray-700">
            Key
            <select
              value={key}
              onChange={(e) => setKey(Number(e.target.value))}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
            >
              {keyOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
        </div>

        <button
          onClick={generateLeadMelody}
          disabled={isLoading || !backingTrackId}
          className={`w-full py-2 px-4 rounded-md text-white font-medium ${
            isLoading || !backingTrackId
              ? "bg-indigo-400 cursor-not-allowed"
              : "bg-indigo-600 hover:bg-indigo-700"
          }`}
        >
          {isLoading ? "Generating..." : "Generate Lead Melody"}
        </button>

        {error && <div className="text-red-600 text-sm mt-2">{error}</div>}

        {!backingTrackId && (
          <div className="text-yellow-600 text-sm mt-2">
            Please generate a backing track first
          </div>
        )}
      </div>
    </div>
  );
};
