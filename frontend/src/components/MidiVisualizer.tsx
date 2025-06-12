import { useEffect, useRef, useState } from "react";
import * as Tone from "tone";
import { Midi } from "@tonejs/midi";

interface MidiVisualizerProps {
  midiUrl: string;
  label: string;
  height?: number;
  width?: number;
}

interface Note {
  note: string;
  midi: number;
  time: number;
  duration: number;
  velocity: number;
}

const MidiVisualizer: React.FC<MidiVisualizerProps> = ({
  midiUrl,
  label,
  height = 200,
  width = 800,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [midiData, setMidiData] = useState<Midi | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const playerRef = useRef<Tone.Player | null>(null);
  const animationRef = useRef<number | null>(null);
  const currentTimeRef = useRef(0); // Direct ref for drawing
  const isPlayingRef = useRef(false); // Direct ref for playing state

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
        setDuration(midi.duration);
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

  const handlePlayPause = async () => {
    if (!midiData) return;

    if (isPlaying) {
      // Stop playback
      if (playerRef.current) {
        playerRef.current.stop();
        playerRef.current.dispose();
        playerRef.current = null;
      }
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
      // Stop and reset transport
      Tone.Transport.stop();
      Tone.Transport.cancel();
      Tone.Transport.position = 0;
      setIsPlaying(false);
      isPlayingRef.current = false; // Set ref immediately
      setCurrentTime(0);
      currentTimeRef.current = 0;
    } else {
      // Start playback
      try {
        // Ensure audio context is started
        await Tone.start();
        console.log("Audio context started");

        // Create synths for different instrument types
        const createSynth = (type: string) => {
          switch (type) {
            case "drums":
              return new Tone.MembraneSynth({
                pitchDecay: 0.05,
                octaves: 10,
                oscillator: { type: "sine" },
                envelope: {
                  attack: 0.001,
                  decay: 0.4,
                  sustain: 0.01,
                  release: 1.4,
                },
              }).toDestination();
            case "bass":
              return new Tone.MonoSynth({
                oscillator: { type: "sawtooth" },
                envelope: {
                  attack: 0.01,
                  decay: 0.2,
                  sustain: 0.4,
                  release: 0.8,
                },
                filterEnvelope: {
                  attack: 0.01,
                  decay: 0.1,
                  sustain: 0.3,
                  release: 0.5,
                },
              }).toDestination();
            default:
              return new Tone.PolySynth(Tone.Synth, {
                oscillator: { type: "triangle" },
                envelope: {
                  attack: 0.01,
                  decay: 0.1,
                  sustain: 0.3,
                  release: 0.5,
                },
              }).toDestination();
          }
        };

        // Create synths for different tracks
        const synths = {
          drums: createSynth("drums"),
          bass: createSynth("bass"),
          melody: createSynth("melody"),
        };

        // Test the synth with a simple note
        console.log("Testing synth with C4...");
        synths.melody.triggerAttackRelease("C4", "4n");

        // Get all notes and sort them by time
        const allNotes: Array<{
          name: string;
          time: number;
          duration: number;
          velocity: number;
          trackIndex: number;
          midi: number;
        }> = [];

        midiData.tracks.forEach((track, trackIndex) => {
          track.notes.forEach((note) => {
            allNotes.push({
              name: note.name,
              time: note.time,
              duration: note.duration,
              velocity: note.velocity,
              trackIndex: trackIndex,
              midi: note.midi,
            });
          });
        });

        console.log("MIDI Debug Info:", {
          totalNotes: allNotes.length,
          tempo: midiData.header.tempos?.[0]?.bpm || 120,
          duration: duration,
          tracks: midiData.tracks.length,
          firstNote: allNotes[0],
          lastNote: allNotes[allNotes.length - 1],
        });

        // Sort notes by time
        allNotes.sort((a, b) => a.time - b.time);

        // Create a set to track which notes have been played
        const playedNotes = new Set<string>();

        // Start playback tracking
        const startTime = Date.now() / 1000; // Use real time instead of Tone.now()

        console.log(
          "Starting playback with startTime:",
          startTime,
          "duration:",
          duration
        );

        setIsPlaying(true);
        isPlayingRef.current = true; // Set ref immediately

        // Start a separate progress update loop for smoother playhead movement
        let frameCount = 0;
        const updateProgress = () => {
          frameCount++;
          console.log("Frame", frameCount, "called");

          if (!isPlayingRef.current) {
            console.log("updateProgress: isPlayingRef is false, stopping");
            return;
          }

          const elapsed = Date.now() / 1000 - startTime;
          currentTimeRef.current = Math.min(elapsed, duration);
          setCurrentTime(currentTimeRef.current);

          console.log(
            "updateProgress: elapsed=",
            elapsed,
            "currentTimeRef.current=",
            currentTimeRef.current,
            "duration=",
            duration
          );

          // Check for notes that should be played at this time
          allNotes.forEach((note) => {
            const noteKey = `${note.name}-${note.time}-${note.duration}-${note.trackIndex}`;
            if (
              !playedNotes.has(noteKey) &&
              elapsed >= note.time &&
              elapsed < note.time + 0.1
            ) {
              console.log(
                "Playing note:",
                note.name,
                "at elapsed:",
                elapsed,
                "velocity:",
                note.velocity,
                "track:",
                note.trackIndex
              );

              // Choose synth based on track index
              let synth;
              if (note.trackIndex === 0) {
                synth = synths.drums; // First track is usually drums
              } else if (note.trackIndex === 1) {
                synth = synths.bass; // Second track is usually bass
              } else {
                synth = synths.melody; // Other tracks are melody
              }

              const durationStr =
                note.duration <= 0.25
                  ? "16n"
                  : note.duration <= 0.5
                  ? "8n"
                  : note.duration <= 1
                  ? "4n"
                  : note.duration <= 2
                  ? "2n"
                  : "1n";

              // For drums, use different notes for different drum sounds
              if (note.trackIndex === 0) {
                // Map MIDI drum notes to different drum sounds
                const drumMap: { [key: number]: string } = {
                  36: "C2", // Bass drum
                  38: "D2", // Snare
                  42: "F#2", // Closed hi-hat
                  46: "A#2", // Open hi-hat
                  49: "C#3", // Crash cymbal
                  51: "D#3", // Ride cymbal
                };
                const drumNote = drumMap[note.midi] || "C2";
                try {
                  synth.triggerAttackRelease(
                    drumNote,
                    durationStr,
                    undefined,
                    note.velocity
                  );
                } catch (err) {
                  console.warn("Failed to play drum note:", drumNote, err);
                }
              } else {
                try {
                  synth.triggerAttackRelease(
                    note.name,
                    durationStr,
                    undefined,
                    note.velocity
                  );
                } catch (err) {
                  console.warn("Failed to play note:", note.name, err);
                }
              }

              playedNotes.add(noteKey);
            }
          });

          // Force canvas redraw
          if (canvasRef.current) {
            const canvas = canvasRef.current;
            const ctx = canvas.getContext("2d");
            if (ctx && midiData) {
              // Clear and redraw
              ctx.clearRect(0, 0, width, height);

              // Get all notes from all tracks
              const allNotes: Note[] = [];
              midiData.tracks.forEach((track) => {
                track.notes.forEach((note) => {
                  allNotes.push({
                    note: note.name,
                    midi: note.midi,
                    time: note.time,
                    duration: note.duration,
                    velocity: note.velocity,
                  });
                });
              });

              if (allNotes.length === 0) {
                ctx.fillStyle = "#666";
                ctx.font = "16px Arial";
                ctx.textAlign = "center";
                ctx.fillText(
                  "No notes found in MIDI file",
                  width / 2,
                  height / 2
                );
                return;
              }

              // Calculate time range
              const maxTime = Math.max(
                ...allNotes.map((n) => n.time + n.duration)
              );
              const minTime = Math.min(...allNotes.map((n) => n.time));
              const timeRange = maxTime - minTime;

              // Calculate note range
              const maxMidi = Math.max(...allNotes.map((n) => n.midi));
              const minMidi = Math.min(...allNotes.map((n) => n.midi));
              const noteRange = maxMidi - minMidi + 1;

              // Set up drawing parameters
              const padding = 40;
              const drawWidth = width - 2 * padding;
              const drawHeight = height - 2 * padding;

              // Draw background
              ctx.fillStyle = "#1a1a1a";
              ctx.fillRect(0, 0, width, height);

              // Draw grid
              ctx.strokeStyle = "#333";
              ctx.lineWidth = 1;

              // Vertical lines (time grid)
              const timeStep = Math.max(1, Math.ceil(timeRange / 10));
              for (let t = 0; t <= timeRange; t += timeStep) {
                const x = padding + (t / timeRange) * drawWidth;
                ctx.beginPath();
                ctx.moveTo(x, padding);
                ctx.lineTo(x, height - padding);
                ctx.stroke();
              }

              // Horizontal lines (note grid)
              const noteStep = Math.max(1, Math.ceil(noteRange / 20));
              for (let n = minMidi; n <= maxMidi; n += noteStep) {
                const y =
                  height - padding - ((n - minMidi) / noteRange) * drawHeight;
                ctx.beginPath();
                ctx.moveTo(padding, y);
                ctx.lineTo(width - padding, y);
                ctx.stroke();
              }

              // Draw notes
              allNotes.forEach((note) => {
                const x =
                  padding + ((note.time - minTime) / timeRange) * drawWidth;
                const y =
                  height -
                  padding -
                  ((note.midi - minMidi) / noteRange) * drawHeight;
                const noteWidth = Math.max(
                  2,
                  (note.duration / timeRange) * drawWidth
                );
                const noteHeight = Math.max(4, drawHeight / noteRange);

                // Check if note is currently playing
                const noteEndTime = note.time + note.duration;
                const isCurrentlyPlaying =
                  isPlayingRef.current &&
                  currentTimeRef.current >= note.time &&
                  currentTimeRef.current <= noteEndTime;

                // Color based on velocity and playing state
                const intensity = Math.floor(note.velocity * 255);
                if (isCurrentlyPlaying) {
                  ctx.fillStyle = `rgb(255, ${intensity}, ${intensity})`; // Red when playing
                } else {
                  ctx.fillStyle = `rgb(${intensity}, ${intensity}, 255)`; // Blue normally
                }

                ctx.fillRect(x, y - noteHeight / 2, noteWidth, noteHeight);

                // Add border
                ctx.strokeStyle = isCurrentlyPlaying ? "#fff" : "#888";
                ctx.lineWidth = isCurrentlyPlaying ? 2 : 1;
                ctx.strokeRect(x, y - noteHeight / 2, noteWidth, noteHeight);
              });

              // Draw playhead
              if (isPlayingRef.current && duration > 0) {
                const playheadTime = currentTimeRef.current;
                const playheadX =
                  padding + ((playheadTime - minTime) / timeRange) * drawWidth;
                ctx.strokeStyle = "#ff4444";
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.moveTo(playheadX, padding);
                ctx.lineTo(playheadX, height - padding);
                ctx.stroke();

                console.log(
                  "Drawing playhead at:",
                  playheadX,
                  "playheadTime:",
                  playheadTime,
                  "minTime:",
                  minTime,
                  "timeRange:",
                  timeRange
                );
              }

              // Draw labels
              ctx.fillStyle = "#fff";
              ctx.font = "12px Arial";
              ctx.textAlign = "left";

              // Time labels
              for (let t = 0; t <= timeRange; t += timeStep) {
                const x = padding + (t / timeRange) * drawWidth;
                ctx.fillText(`${t.toFixed(1)}s`, x + 2, height - 5);
              }

              // Note labels (show every 12 notes for octaves)
              for (let n = minMidi; n <= maxMidi; n += 12) {
                const y =
                  height - padding - ((n - minMidi) / noteRange) * drawHeight;
                const noteName = Tone.Frequency(n, "midi").toNote();
                ctx.fillText(noteName, 5, y + 4);
              }
            }
          }

          if (elapsed < duration) {
            console.log(
              "Scheduling next frame, elapsed:",
              elapsed,
              "duration:",
              duration
            );
            requestAnimationFrame(updateProgress);
          } else {
            console.log("Playback finished");
            setIsPlaying(false);
            isPlayingRef.current = false;
            setCurrentTime(0);
            currentTimeRef.current = 0;
          }
        };

        // Start the progress loop
        updateProgress();
      } catch (err) {
        console.error("Error playing MIDI:", err);
        setError("Failed to play MIDI file");
      }
    }
  };

  // Simple audio test function
  const testAudio = async () => {
    try {
      await Tone.start();
      const synth = new Tone.Synth().toDestination();
      synth.triggerAttackRelease("C4", "4n");
      console.log("Audio test completed - you should hear a C4 note");
    } catch (err) {
      console.error("Audio test failed:", err);
    }
  };

  useEffect(() => {
    if (!midiData || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Get all notes from all tracks
    const allNotes: Note[] = [];
    midiData.tracks.forEach((track) => {
      track.notes.forEach((note) => {
        allNotes.push({
          note: note.name,
          midi: note.midi,
          time: note.time,
          duration: note.duration,
          velocity: note.velocity,
        });
      });
    });

    if (allNotes.length === 0) {
      // Draw "No notes" message
      ctx.fillStyle = "#666";
      ctx.font = "16px Arial";
      ctx.textAlign = "center";
      ctx.fillText("No notes found in MIDI file", width / 2, height / 2);
      return;
    }

    // Calculate time range
    const maxTime = Math.max(...allNotes.map((n) => n.time + n.duration));
    const minTime = Math.min(...allNotes.map((n) => n.time));
    const timeRange = maxTime - minTime;

    // Calculate note range
    const maxMidi = Math.max(...allNotes.map((n) => n.midi));
    const minMidi = Math.min(...allNotes.map((n) => n.midi));
    const noteRange = maxMidi - minMidi + 1;

    // Set up drawing parameters
    const padding = 40;
    const drawWidth = width - 2 * padding;
    const drawHeight = height - 2 * padding;

    // Draw background
    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 1;

    // Vertical lines (time grid)
    const timeStep = Math.max(1, Math.ceil(timeRange / 10));
    for (let t = 0; t <= timeRange; t += timeStep) {
      const x = padding + (t / timeRange) * drawWidth;
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, height - padding);
      ctx.stroke();
    }

    // Horizontal lines (note grid)
    const noteStep = Math.max(1, Math.ceil(noteRange / 20));
    for (let n = minMidi; n <= maxMidi; n += noteStep) {
      const y = height - padding - ((n - minMidi) / noteRange) * drawHeight;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }

    // Draw notes
    allNotes.forEach((note) => {
      const x = padding + ((note.time - minTime) / timeRange) * drawWidth;
      const y =
        height - padding - ((note.midi - minMidi) / noteRange) * drawHeight;
      const noteWidth = Math.max(2, (note.duration / timeRange) * drawWidth);
      const noteHeight = Math.max(4, drawHeight / noteRange);

      // Check if note is currently playing
      const noteEndTime = note.time + note.duration;
      const isCurrentlyPlaying =
        isPlayingRef.current &&
        currentTimeRef.current >= note.time &&
        currentTimeRef.current <= noteEndTime;

      // Color based on velocity and playing state
      const intensity = Math.floor(note.velocity * 255);
      if (isCurrentlyPlaying) {
        ctx.fillStyle = `rgb(255, ${intensity}, ${intensity})`; // Red when playing
      } else {
        ctx.fillStyle = `rgb(${intensity}, ${intensity}, 255)`; // Blue normally
      }

      ctx.fillRect(x, y - noteHeight / 2, noteWidth, noteHeight);

      // Add border
      ctx.strokeStyle = isCurrentlyPlaying ? "#fff" : "#888";
      ctx.lineWidth = isCurrentlyPlaying ? 2 : 1;
      ctx.strokeRect(x, y - noteHeight / 2, noteWidth, noteHeight);
    });

    // Draw playhead
    if (isPlayingRef.current && duration > 0) {
      const playheadTime = currentTimeRef.current;
      const playheadX =
        padding + ((playheadTime - minTime) / timeRange) * drawWidth;
      ctx.strokeStyle = "#ff4444";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(playheadX, padding);
      ctx.lineTo(playheadX, height - padding);
      ctx.stroke();

      console.log(
        "Drawing playhead at:",
        playheadX,
        "playheadTime:",
        playheadTime,
        "minTime:",
        minTime,
        "timeRange:",
        timeRange
      );
    }

    // Draw labels
    ctx.fillStyle = "#fff";
    ctx.font = "12px Arial";
    ctx.textAlign = "left";

    // Time labels
    for (let t = 0; t <= timeRange; t += timeStep) {
      const x = padding + (t / timeRange) * drawWidth;
      ctx.fillText(`${t.toFixed(1)}s`, x + 2, height - 5);
    }

    // Note labels (show every 12 notes for octaves)
    for (let n = minMidi; n <= maxMidi; n += 12) {
      const y = height - padding - ((n - minMidi) / noteRange) * drawHeight;
      const noteName = Tone.Frequency(n, "midi").toNote();
      ctx.fillText(noteName, 5, y + 4);
    }
  }, [midiData, width, height, isPlaying, currentTime, duration]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (playerRef.current) {
        playerRef.current.dispose();
      }
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      // Clean up transport
      Tone.Transport.stop();
      Tone.Transport.cancel();
      Tone.Transport.position = 0;
    };
  }, []);

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-lg font-semibold mb-2 text-blue-300">{label}</h3>
        <div className="flex items-center justify-center h-32">
          <div className="text-gray-400">Loading MIDI visualization...</div>
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
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-lg font-semibold text-blue-300">{label}</h3>
        <div className="flex gap-2">
          <button
            onClick={testAudio}
            className="px-2 py-1 rounded text-xs font-medium bg-yellow-600 hover:bg-yellow-700 text-white"
          >
            🔊 Test Audio
          </button>
          <button
            onClick={handlePlayPause}
            disabled={!midiData}
            className={`px-3 py-1 rounded text-sm font-medium transition-colors
              ${
                !midiData
                  ? "bg-gray-600 text-gray-400 cursor-not-allowed"
                  : isPlaying
                  ? "bg-red-600 hover:bg-red-700 text-white"
                  : "bg-green-600 hover:bg-green-700 text-white"
              }`}
          >
            {isPlaying ? "⏸️ Pause" : "▶️ Play"}
          </button>
        </div>
      </div>

      <div className="overflow-x-auto">
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          className="border border-gray-600 rounded"
        />
      </div>

      {/* Progress bar */}
      {duration > 0 && (
        <div className="mt-2">
          <div className="flex justify-between text-xs text-gray-400 mb-1">
            <span>{currentTime.toFixed(1)}s</span>
            <span>{duration.toFixed(1)}s</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div
              className="bg-blue-500 h-2 rounded-full transition-all duration-100"
              style={{ width: `${(currentTime / duration) * 100}%` }}
            />
          </div>
        </div>
      )}

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

export default MidiVisualizer;
