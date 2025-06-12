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
      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-lg font-semibold mb-2 text-blue-300">{label}</h3>
        <div className="flex items-center justify-center h-32">
          <div className="text-gray-400">Loading sheet music...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-lg font-semibold mb-2 text-blue-300">{label}</h3>
        <div className="flex items-center justify-center h-32">
          <div className="text-red-400">Error: {error}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <h3 className="text-lg font-semibold mb-2 text-blue-300">{label}</h3>
      <div className="overflow-x-auto">
        <div
          ref={containerRef}
          className="border border-gray-600 rounded bg-white"
          style={{ minHeight: height }}
        />
      </div>
      {midiData && (
        <div className="mt-2 text-sm text-gray-400">
          <span>Tracks: {midiData.tracks.length}</span>
          <span className="mx-2">•</span>
          <span>Duration: {midiData.duration.toFixed(1)}s</span>
          <span className="mx-2">•</span>
          <span>
            Notes:{" "}
            {midiData.tracks.reduce(
              (sum, track) => sum + track.notes.length,
              0
            )}
          </span>
        </div>
      )}
    </div>
  );
};

export default SheetMusicVisualizer;
