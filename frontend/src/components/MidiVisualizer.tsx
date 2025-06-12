import React, { useEffect, useRef, useState } from "react";
import { Midi } from "@tonejs/midi";
import * as Tone from "tone";

interface MidiVisualizerProps {
  midiUrl: string;
  label: string;
  height?: number;
  width?: number;
}

const MidiVisualizer: React.FC<MidiVisualizerProps> = ({
  midiUrl,
  label,
  height = 300,
  width = 800,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [midiData, setMidiData] = useState<Midi | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [samplers, setSamplers] = useState<{ [key: number]: any }>({});
  const [samplersLoaded, setSamplersLoaded] = useState(false);
  const [midiTempo, setMidiTempo] = useState(120); // Add tempo state

  // Use refs to track state that needs to be accessed in the update loop
  const isPlayingRef = useRef(false);
  const currentTimeRef = useRef(0);

  // Initialize soundfont samplers
  useEffect(() => {
    const initializeSamplers = async () => {
      try {
        console.log("Initializing instruments...");

        // Create instruments for different types
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const newSamplers: { [key: number]: any } = {};

        // Piano (program 0) - Main instrument
        newSamplers[0] = new Tone.PolySynth(Tone.Synth, {
          oscillator: { type: "triangle" },
          envelope: {
            attack: 0.01,
            decay: 0.1,
            sustain: 0.3,
            release: 0.5,
          },
          volume: 20,
        }).toDestination();

        // Bass (program 32) - Use monophonic synth
        newSamplers[32] = new Tone.MonoSynth({
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
          volume: 18,
        }).toDestination();

        // Lead synth (program 81) - Use polyphonic synth
        newSamplers[81] = new Tone.PolySynth(Tone.Synth, {
          oscillator: { type: "square" },
          envelope: {
            attack: 0.01,
            decay: 0.1,
            sustain: 0.3,
            release: 0.5,
          },
          volume: 16,
        }).toDestination();

        // Drums (track 0) - Use membrane synth
        newSamplers[-1] = new Tone.MembraneSynth({
          pitchDecay: 0.05,
          octaves: 10,
          oscillator: { type: "sine" },
          envelope: {
            attack: 0.001,
            decay: 0.4,
            sustain: 0.01,
            release: 1.4,
          },
          volume: 22,
        }).toDestination();

        console.log("Instruments created:", Object.keys(newSamplers));
        setSamplers(newSamplers);
        setSamplersLoaded(true);
        console.log("All instruments initialized and ready");
      } catch (error) {
        console.error("Error initializing instruments:", error);
        // Fallback to basic synths if samplers fail
        setSamplersLoaded(true);
      }
    };

    initializeSamplers();
  }, []);

  // Load MIDI data
  useEffect(() => {
    const loadMidi = async () => {
      try {
        setIsLoading(true);
        setError(null);
        console.log("Loading MIDI from:", midiUrl);

        const response = await fetch(midiUrl);
        if (!response.ok) {
          throw new Error(`Failed to fetch MIDI: ${response.statusText}`);
        }

        const arrayBuffer = await response.arrayBuffer();
        const midi = new Midi(arrayBuffer);

        console.log("MIDI loaded successfully:", midi);
        console.log("Tracks:", midi.tracks.length);
        midi.tracks.forEach((track, i) => {
          console.log(`Track ${i}:`, track.name, "notes:", track.notes.length);
        });

        // Extract tempo from MIDI data
        let tempo = 120; // Default tempo
        if (
          midi.header &&
          midi.header.tempos &&
          midi.header.tempos.length > 0
        ) {
          // Convert microseconds per quarter note to BPM
          const microsecondsPerBeat = midi.header.tempos[0].bpm;
          tempo = Math.round(microsecondsPerBeat);
          console.log("Extracted MIDI tempo:", tempo, "BPM");
        } else {
          console.log("No tempo found in MIDI, using default 120 BPM");
        }

        setMidiData(midi);
        setMidiTempo(tempo);
        setDuration(midi.duration);
        setIsLoading(false);
      } catch (err) {
        console.error("Error loading MIDI:", err);
        setError(err instanceof Error ? err.message : "Failed to load MIDI");
        setIsLoading(false);
      }
    };

    if (midiUrl) {
      loadMidi();
    }
  }, [midiUrl]);

  // Draw piano roll
  useEffect(() => {
    if (!midiData || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Set background
    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 1;

    // Vertical lines (time)
    const timeStep = width / 16; // 16 time divisions
    for (let i = 0; i <= 16; i++) {
      ctx.beginPath();
      ctx.moveTo(i * timeStep, 0);
      ctx.lineTo(i * timeStep, height);
      ctx.stroke();
    }

    // Horizontal lines (notes)
    const noteStep = height / 48; // 4 octaves
    for (let i = 0; i <= 48; i++) {
      ctx.beginPath();
      ctx.moveTo(0, i * noteStep);
      ctx.lineTo(width, i * noteStep);
      ctx.stroke();
    }

    // Draw notes
    const allNotes: Array<{
      name: string;
      time: number;
      duration: number;
      velocity: number;
      trackIndex: number;
      midi: number;
      program: number;
    }> = [];

    midiData.tracks.forEach((track, trackIndex) => {
      track.notes.forEach((note) => {
        allNotes.push({
          name: note.name,
          time: note.time,
          duration: note.duration,
          velocity: note.velocity,
          trackIndex,
          midi: note.midi,
          program: 0, // Default to piano, we'll determine by track index
        });
      });
    });

    // Sort notes by time
    allNotes.sort((a, b) => a.time - b.time);

    // Draw each note
    allNotes.forEach((note) => {
      const x = (note.time / duration) * width;
      const noteHeight = height / 48;
      const y = height - (note.midi - 36) * noteHeight; // Start from C2 (MIDI 36)
      const noteWidth = Math.max(2, (note.duration / duration) * width);

      // Color based on track
      const colors = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#feca57"];
      const color = colors[note.trackIndex % colors.length];

      // Highlight if currently playing
      const isCurrentlyPlaying =
        currentTime >= note.time && currentTime <= note.time + note.duration;

      ctx.fillStyle = isCurrentlyPlaying ? "#ffff00" : color;
      ctx.fillRect(x, y, noteWidth, noteHeight - 1);

      // Add border
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y, noteWidth, noteHeight - 1);
    });

    // Draw playhead
    if (isPlaying) {
      const playheadX = (currentTime / duration) * width;
      ctx.strokeStyle = "#ff0000";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(playheadX, 0);
      ctx.lineTo(playheadX, height);
      ctx.stroke();
    }
  }, [midiData, currentTime, isPlaying, duration, width, height]);

  const handlePlayPause = async () => {
    console.log(
      "Play button clicked, midiData:",
      !!midiData,
      "samplersLoaded:",
      samplersLoaded
    );

    if (!midiData || !samplersLoaded) {
      console.log("Cannot play - missing data or samplers not loaded");
      return;
    }

    if (isPlayingRef.current) {
      // Stop playback
      console.log("Stopping playback");
      isPlayingRef.current = false;
      setIsPlaying(false);
      setCurrentTime(0);
      currentTimeRef.current = 0;
      Tone.Transport.stop();
      Tone.Transport.cancel();
      return;
    }

    // Start playback
    try {
      console.log("Starting playback...");
      await Tone.start();
      console.log("Audio context started");

      isPlayingRef.current = true;
      setIsPlaying(true);
      setCurrentTime(0);
      currentTimeRef.current = 0;

      // Get all notes and sort them by time
      const allNotes: Array<{
        name: string;
        time: number;
        duration: number;
        velocity: number;
        trackIndex: number;
        midi: number;
        program: number;
      }> = [];

      midiData.tracks.forEach((track, trackIndex) => {
        track.notes.forEach((note) => {
          allNotes.push({
            name: note.name,
            time: note.time,
            duration: note.duration,
            velocity: note.velocity,
            trackIndex,
            midi: note.midi,
            program: 0, // Default to piano, we'll determine by track index
          });
        });
      });

      allNotes.sort((a, b) => a.time - b.time);
      console.log("Total notes to play:", allNotes.length);

      // Debug: show first few notes
      console.log(
        "First 5 notes:",
        allNotes.slice(0, 5).map((n) => ({
          name: n.name,
          time: n.time,
          track: n.trackIndex,
          velocity: n.velocity,
        }))
      );

      const playedNotes = new Set<string>();
      const startTime = Date.now();

      const updateProgress = () => {
        if (!isPlayingRef.current) {
          console.log("Playback stopped, exiting update loop");
          return;
        }

        const elapsed = (Date.now() - startTime) / 1000;
        currentTimeRef.current = elapsed;
        setCurrentTime(elapsed);

        // Debug: log every second
        if (Math.floor(elapsed) !== Math.floor(currentTimeRef.current - 0.1)) {
          console.log(
            "Playback progress:",
            elapsed.toFixed(1),
            "s at",
            midiTempo,
            "BPM"
          );
        }

        // Check for notes to play
        allNotes.forEach((note) => {
          const noteKey = `${note.trackIndex}-${note.midi}-${note.time}`;

          if (
            !playedNotes.has(noteKey) &&
            elapsed >= note.time &&
            elapsed < note.time + 0.1
          ) {
            // Choose sampler based on program number
            let sampler;
            if (note.trackIndex === 0) {
              sampler = samplers[-1]; // First track is usually drums
            } else if (note.trackIndex === 1) {
              sampler = samplers[32]; // Second track is usually bass
            } else {
              sampler = samplers[note.program] || samplers[0]; // Use piano as fallback
            }

            if (sampler) {
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

              console.log("About to trigger note:", {
                note: note.name,
                duration: durationStr,
                velocity: note.velocity / 127,
                sampler: sampler.constructor.name,
              });

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
                  console.log("Triggering drum note:", drumNote);
                  // Use immediate timing to avoid conflicts
                  sampler.triggerAttackRelease(
                    drumNote,
                    durationStr,
                    undefined,
                    Math.max(0.1, note.velocity / 127) // Minimum volume of 0.1
                  );
                  console.log("Drum note triggered successfully");
                } catch (err) {
                  console.warn("Failed to play drum note:", drumNote, err);
                }
              } else {
                try {
                  console.log("Triggering melodic note:", note.name);
                  // Use immediate timing to avoid conflicts
                  sampler.triggerAttackRelease(
                    note.name,
                    durationStr,
                    undefined,
                    Math.max(0.1, note.velocity / 127) // Minimum volume of 0.1
                  );
                  console.log("Melodic note triggered successfully");
                } catch (err) {
                  console.warn("Failed to play note:", note.name, err);
                }
              }
            } else {
              console.warn("No sampler found for track:", note.trackIndex);
            }

            playedNotes.add(noteKey);
          }
        });

        if (elapsed < duration) {
          requestAnimationFrame(updateProgress);
        } else {
          console.log("Playback finished");
          isPlayingRef.current = false;
          setIsPlaying(false);
          setCurrentTime(0);
          currentTimeRef.current = 0;
        }
      };

      console.log("Starting update loop...");
      updateProgress();
    } catch (error) {
      console.error("Error starting playback:", error);
      isPlayingRef.current = false;
      setIsPlaying(false);
    }
  };

  // Handle click-to-seek on progress bar
  const handleProgressBarClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!duration) return;

    const rect = event.currentTarget.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const percentage = clickX / rect.width;
    const newTime = percentage * duration;

    setCurrentTime(newTime);
    currentTimeRef.current = newTime;

    // If playing, restart from new position
    if (isPlayingRef.current) {
      // Stop current playback
      isPlayingRef.current = false;
      setIsPlaying(false);

      // Restart playback from new position
      setTimeout(() => {
        startPlaybackFromTime(newTime);
      }, 100);
    }
  };

  // Handle click-to-seek on piano roll
  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!duration) return;

    const rect = event.currentTarget.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const percentage = clickX / rect.width;
    const newTime = percentage * duration;

    setCurrentTime(newTime);
    currentTimeRef.current = newTime;

    // If playing, restart from new position
    if (isPlayingRef.current) {
      // Stop current playback
      isPlayingRef.current = false;
      setIsPlaying(false);

      // Restart playback from new position
      setTimeout(() => {
        startPlaybackFromTime(newTime);
      }, 100);
    }
  };

  // Function to start playback from a specific time
  const startPlaybackFromTime = async (startTime: number) => {
    if (!midiData || !samplersLoaded) return;

    try {
      console.log("Starting playback from time:", startTime);
      await Tone.start();

      isPlayingRef.current = true;
      setIsPlaying(true);
      setCurrentTime(startTime);
      currentTimeRef.current = startTime;

      // Get all notes and sort them by time
      const allNotes: Array<{
        name: string;
        time: number;
        duration: number;
        velocity: number;
        trackIndex: number;
        midi: number;
        program: number;
      }> = [];

      midiData.tracks.forEach((track, trackIndex) => {
        track.notes.forEach((note) => {
          allNotes.push({
            name: note.name,
            time: note.time,
            duration: note.duration,
            velocity: note.velocity,
            trackIndex,
            midi: note.midi,
            program: 0,
          });
        });
      });

      allNotes.sort((a, b) => a.time - b.time);

      const playedNotes = new Set<string>();
      const playbackStartTime = Date.now() - startTime * 1000; // Adjust for seek time

      const updateProgress = () => {
        if (!isPlayingRef.current) {
          console.log("Playback stopped, exiting update loop");
          return;
        }

        const elapsed = (Date.now() - playbackStartTime) / 1000;
        currentTimeRef.current = elapsed;
        setCurrentTime(elapsed);

        // Check for notes to play
        allNotes.forEach((note) => {
          const noteKey = `${note.trackIndex}-${note.midi}-${note.time}`;

          if (
            !playedNotes.has(noteKey) &&
            elapsed >= note.time &&
            elapsed < note.time + 0.1
          ) {
            // Choose sampler based on program number
            let sampler;
            if (note.trackIndex === 0) {
              sampler = samplers[-1]; // First track is usually drums
            } else if (note.trackIndex === 1) {
              sampler = samplers[32]; // Second track is usually bass
            } else {
              sampler = samplers[note.program] || samplers[0]; // Use piano as fallback
            }

            if (sampler) {
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
                  sampler.triggerAttackRelease(
                    drumNote,
                    durationStr,
                    undefined,
                    Math.max(0.1, note.velocity / 127)
                  );
                } catch (err) {
                  console.warn("Failed to play drum note:", drumNote, err);
                }
              } else {
                try {
                  sampler.triggerAttackRelease(
                    note.name,
                    durationStr,
                    undefined,
                    Math.max(0.1, note.velocity / 127)
                  );
                } catch (err) {
                  console.warn("Failed to play note:", note.name, err);
                }
              }
            }

            playedNotes.add(noteKey);
          }
        });

        if (elapsed < duration) {
          requestAnimationFrame(updateProgress);
        } else {
          console.log("Playback finished");
          isPlayingRef.current = false;
          setIsPlaying(false);
          setCurrentTime(0);
          currentTimeRef.current = 0;
        }
      };

      updateProgress();
    } catch (error) {
      console.error("Error starting playback from time:", error);
      isPlayingRef.current = false;
      setIsPlaying(false);
    }
  };

  if (isLoading) {
    return (
      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-lg font-semibold mb-2 text-blue-300">{label}</h3>
        <div className="flex items-center justify-center h-32">
          <div className="text-gray-400">Loading MIDI data...</div>
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
            onClick={handlePlayPause}
            disabled={!midiData || !samplersLoaded}
            className={`px-3 py-1 rounded text-sm font-medium transition-colors
              ${
                !midiData || !samplersLoaded
                  ? "bg-gray-600 text-gray-400 cursor-not-allowed"
                  : isPlaying
                  ? "bg-red-600 hover:bg-red-700 text-white"
                  : "bg-green-600 hover:bg-green-700 text-white"
              }`}
          >
            {isPlaying ? "⏸️ Pause" : "▶️ Play"}
          </button>
          {!samplersLoaded && (
            <span className="text-xs text-gray-400">
              Loading instruments...
            </span>
          )}
        </div>
      </div>

      {/* Progress bar */}
      {duration > 0 && (
        <div className="mb-2">
          <div className="flex justify-between text-xs text-gray-400 mb-1">
            <span>{currentTime.toFixed(1)}s</span>
            <span>{duration.toFixed(1)}s</span>
          </div>
          <div
            className="w-full bg-gray-700 rounded-full h-2 cursor-pointer"
            onClick={handleProgressBarClick}
          >
            <div
              className="bg-blue-500 h-2 rounded-full transition-all duration-100"
              style={{ width: `${(currentTime / duration) * 100}%` }}
            />
          </div>
        </div>
      )}

      <div className="overflow-x-auto">
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          className="border border-gray-600 rounded cursor-pointer"
          onClick={handleCanvasClick}
        />
      </div>

      {midiData && (
        <div className="mt-2 text-sm text-gray-400">
          <span>Tracks: {midiData.tracks.length}</span>
          <span className="mx-2">•</span>
          <span>Duration: {midiData.duration.toFixed(1)}s</span>
          <span className="mx-2">•</span>
          <span>Tempo: {midiTempo} BPM</span>
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
