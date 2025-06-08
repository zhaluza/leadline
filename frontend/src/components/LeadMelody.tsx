import { useState } from "react";

interface LeadMelodyProps {
  generationId: string | null;
  onGenerationComplete: (audioUrl: string) => void;
  onError: (error: string) => void;
}

export default function LeadMelody({
  generationId,
  onGenerationComplete,
  onError,
}: LeadMelodyProps) {
  const [isGenerating, setIsGenerating] = useState(false);

  const handleGenerate = async () => {
    if (!generationId) {
      onError("Please generate a backing track first");
      return;
    }

    setIsGenerating(true);
    try {
      const response = await fetch(
        `http://localhost:8000/api/lead/${generationId}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            num_bars: 8,
            tempo: 120,
          }),
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to generate lead melody");
      }

      const data = await response.json();
      onGenerationComplete(`http://localhost:8000${data.audio_url}`);
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
    <div className="p-4 bg-white rounded-lg shadow-md">
      <h2 className="text-xl font-bold mb-4">Lead Melody Generator</h2>

      <div className="space-y-4">
        <button
          onClick={handleGenerate}
          disabled={isGenerating || !generationId}
          className={`w-full py-2 px-4 rounded-lg font-medium transition-colors
            ${
              !generationId
                ? "bg-gray-700 text-gray-400 cursor-not-allowed"
                : isGenerating
                ? "bg-gray-700 text-gray-300 cursor-not-allowed"
                : "bg-indigo-800 hover:bg-indigo-900 text-white shadow-md hover:shadow-lg"
            }`}
        >
          {!generationId
            ? "Generate Backing Track First"
            : isGenerating
            ? "Generating..."
            : "Generate Lead Melody"}
        </button>

        {generationId && (
          <div className="text-yellow-600 text-sm mt-2">
            Please generate a backing track first
          </div>
        )}
      </div>
    </div>
  );
}
