import { useEffect, useRef, useState } from "react";
import { Midi } from "@tonejs/midi";
import { Factory } from "vexflow";

interface SheetMusicVisualizerProps {
  midiUrl: string;
  label: string;
  height?: number;
  width?: number;
}

const SheetMusicVisualizer: React.FC<SheetMusicVisualizerProps> = ({
  midiUrl,
  label,
  height = 300,
  width = 800,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [midiData, setMidiData] = useState<Midi | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadMidi = async () => {
      try {
        setLoading(true);
        setError(null);

        const response = await fetch(midiUrl);
        if (!response.ok) {
          throw new Error(`Failed to load MIDI file: ${response.statusText}`);
        }

        const arrayBuffer = await response.arrayBuffer();
        const midi = new Midi(arrayBuffer);
        setMidiData(midi);
      } catch (err) {
        console.error("Error loading MIDI:", err);
        setError(
          err instanceof Error ? err.message : "Failed to load MIDI file"
        );
      } finally {
        setLoading(false);
      }
    };

    if (midiUrl) {
      loadMidi();
    }
  }, [midiUrl]);

  useEffect(() => {
    if (!midiData || !containerRef.current) return;

    // Clear previous content
    containerRef.current.innerHTML = "";

    try {
      // Create a unique ID for the VexFlow renderer
      const elementId = `vexflow-${Date.now()}`;
      const div = document.createElement("div");
      div.id = elementId;
      containerRef.current.appendChild(div);

      // Initialize VexFlow
      const factory = new Factory({
        renderer: { elementId, width, height },
      });

      const score = factory.EasyScore();
      const system = factory.System();

      // Convert MIDI notes to VexFlow format
      const notes: string[] = [];
      const durations: string[] = [];

      // Sort notes by time
      const allNotes: Array<{ note: string; time: number; duration: number }> =
        [];
      midiData.tracks.forEach((track) => {
        track.notes.forEach((note) => {
          allNotes.push({
            note: note.name,
            time: note.time,
            duration: note.duration,
          });
        });
      });

      allNotes.sort((a, b) => a.time - b.time);

      // Convert to VexFlow notation (simplified)
      allNotes.slice(0, 16).forEach((note) => {
        // Limit to first 16 notes for demo
        notes.push(note.note);
        // Convert duration to VexFlow format
        if (note.duration <= 0.25) {
          durations.push("16");
        } else if (note.duration <= 0.5) {
          durations.push("8");
        } else if (note.duration <= 1) {
          durations.push("4");
        } else if (note.duration <= 2) {
          durations.push("2");
        } else {
          durations.push("1");
        }
      });

      if (notes.length === 0) {
        // Show "No notes" message
        div.style.textAlign = "center";
        div.style.padding = "20px";
        div.style.color = "#666";
        div.textContent = "No notes found in MIDI file";
        return;
      }

      // Create the score
      system
        .addStave({
          voices: [score.voice(score.notes(notes.join(", "), { stem: "up" }))],
        })
        .addClef("treble")
        .addTimeSignature("4/4");

      factory.draw();
    } catch (err) {
      console.error("Error rendering sheet music:", err);
      setError("Failed to render sheet music");
    }
  }, [midiData, width, height]);

  if (loading) {
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <div className="flex-shrink-0 w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center">
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
                d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"
              />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-blue-300">{label}</h3>
        </div>

        <div className="flex items-center justify-center h-32 bg-gray-700/30 rounded-xl border border-gray-600/50">
          <div className="flex items-center gap-3 text-gray-400">
            <div className="w-5 h-5 border-2 border-gray-400/30 border-t-gray-400 rounded-full animate-spin"></div>
            <span>Loading sheet music...</span>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <div className="flex-shrink-0 w-6 h-6 bg-red-600 rounded-full flex items-center justify-center">
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
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-red-300">{label}</h3>
        </div>

        <div className="flex items-center justify-center h-32 bg-red-900/20 rounded-xl border border-red-500/30">
          <div className="text-red-400 font-medium">Error: {error}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <div className="flex-shrink-0 w-6 h-6 bg-green-600 rounded-full flex items-center justify-center">
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
              d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"
            />
          </svg>
        </div>
        <h3 className="text-lg font-semibold text-green-300">{label}</h3>
      </div>

      <div className="overflow-x-auto">
        <div
          ref={containerRef}
          className="border border-gray-600/50 rounded-xl bg-white shadow-lg"
          style={{ minHeight: height }}
        />
      </div>

      {midiData && (
        <div className="flex flex-wrap gap-4 text-sm text-gray-400 bg-gray-700/30 rounded-lg p-3 border border-gray-600/50">
          <div className="flex items-center gap-1">
            <svg
              className="h-4 w-4"
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
            <span>Tracks: {midiData.tracks.length}</span>
          </div>
          <div className="flex items-center gap-1">
            <svg
              className="h-4 w-4"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <span>Duration: {midiData.duration.toFixed(1)}s</span>
          </div>
          <div className="flex items-center gap-1">
            <svg
              className="h-4 w-4"
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
            <span>
              Notes:{" "}
              {midiData.tracks.reduce(
                (sum, track) => sum + track.notes.length,
                0
              )}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default SheetMusicVisualizer;
