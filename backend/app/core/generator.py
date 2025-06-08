"""
AMT Backing Generator Core Module
================================

This module provides the core AMTBackingGenerator class for generating backing tracks
and lead melodies. This is the same implementation as the original, but moved to the
core directory for the web application.
"""

import torch
from transformers import AutoModelForCausalLM
from anticipation.sample import generate
from anticipation.convert import events_to_midi, midi_to_events
import pretty_midi
import numpy as np
from IPython.display import Audio
import os
from pathlib import Path
import subprocess
import tempfile
import re

class AMTBackingGenerator:
    def __init__(self, model_name='stanford-crfm/music-medium-800k', soundfont_path=None):
        """Initialize the AMT backing generator with the specified model and soundfont."""
        print("Loading AMT model...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model = self.model.to(self.device)
        print(f"Model loaded on {self.device}")
        
        # Check if fluidsynth is installed
        try:
            result = subprocess.run(['fluidsynth', '--version'], 
                                 capture_output=True, 
                                 text=True, 
                                 check=False)
            if result.returncode == 0:
                print(f"Fluidsynth found: {result.stdout.strip()}")
            else:
                print("Warning: fluidsynth command failed. Please ensure it's installed correctly.")
                print(f"Error: {result.stderr}")
        except FileNotFoundError:
            print("Warning: fluidsynth not found. Please install it:")
            print("  macOS: brew install fluid-synth")
            print("  Ubuntu/Debian: sudo apt-get install fluidsynth")
            print("  Windows: Download from https://www.fluidsynth.org/")
        
        # Set up soundfont
        self.soundfont_path = soundfont_path
        if soundfont_path is None:
            # Try to find a default soundfont
            possible_paths = [
                "/usr/share/sounds/sf2/FluidR3_GM.sf2",  # Linux
                "/usr/local/share/fluidsynth/sf2/FluidR3_GM.sf2",  # macOS
                "C:/soundfonts/FluidR3_GM.sf2",  # Windows
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    self.soundfont_path = path
                    break
        
        if self.soundfont_path is None:
            print("Warning: No soundfont found. MIDI playback will use default synthesizer.")
        else:
            print(f"Using soundfont: {self.soundfont_path}")
            if not os.path.exists(self.soundfont_path):
                print(f"Warning: Soundfont file not found at {self.soundfont_path}")
        
        # Create output directory if it doesn't exist
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

    def midi_to_audio(self, midi_file, sample_rate=44100):
        """Convert MIDI to audio using fluidsynth and the specified soundfont."""
        if self.soundfont_path is None:
            print("No soundfont specified, falling back to pretty_midi synthesizer...")
            return midi_file.synthesize(fs=sample_rate)
        
        if not os.path.exists(self.soundfont_path):
            print(f"Soundfont not found at {self.soundfont_path}, falling back to pretty_midi synthesizer...")
            return midi_file.synthesize(fs=sample_rate)
        
        # Create a temporary file for the audio output
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        try:
            # Save the MIDI file to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as temp_midi:
                temp_midi_path = temp_midi.name
                midi_file.write(temp_midi_path)
                print(f"Saved temporary MIDI file to {temp_midi_path}")
            
            # Use fluidsynth to convert MIDI to audio
            cmd = [
                'fluidsynth',
                '-F', temp_audio_path,  # Output file
                '-r', str(sample_rate),  # Sample rate
                '-g', '1.0',  # Gain
                '-i',  # Input file
                temp_midi_path,  # MIDI file
                self.soundfont_path,  # Soundfont
                '-q'  # Quiet mode
            ]
            
            print(f"Running FluidSynth command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if result.returncode != 0:
                print(f"Fluidsynth error output: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
            
            print(f"FluidSynth completed successfully. Output file: {temp_audio_path}")
            
            # Verify the output file exists and has content
            if not os.path.exists(temp_audio_path):
                raise FileNotFoundError(f"FluidSynth did not create output file at {temp_audio_path}")
            
            file_size = os.path.getsize(temp_audio_path)
            print(f"Generated audio file size: {file_size} bytes")
            
            if file_size == 0:
                raise ValueError("FluidSynth generated an empty audio file")
            
            # Read the generated audio file
            audio_data = np.frombuffer(open(temp_audio_path, 'rb').read(), dtype=np.int16)
            print(f"Read audio data: shape={audio_data.shape}, dtype={audio_data.dtype}")
            
            # Convert to float32 and normalize
            audio_data = audio_data.astype(np.float32) / 32768.0
            print(f"Normalized audio data: min={np.min(audio_data)}, max={np.max(audio_data)}")
            
            return audio_data
            
        except subprocess.CalledProcessError as e:
            print(f"Error running fluidsynth: {e}")
            print(f"Command output: {e.stdout}")
            print(f"Error output: {e.stderr}")
            print("Falling back to pretty_midi synthesizer...")
            return midi_file.synthesize(fs=sample_rate)
        except Exception as e:
            print(f"Unexpected error: {e}")
            print("Falling back to pretty_midi synthesizer...")
            return midi_file.synthesize(fs=sample_rate)
        finally:
            # Clean up temporary files
            try:
                os.unlink(temp_audio_path)
                os.unlink(temp_midi_path)
                print("Cleaned up temporary files")
            except Exception as e:
                print(f"Error cleaning up temporary files: {e}")

    def _get_num_bars_for_duration(self, duration_seconds, tempo, beats_per_bar=4):
        """Helper to calculate the number of bars needed for a given duration and tempo."""
        seconds_per_bar = 60.0 / tempo * beats_per_bar
        return int(np.ceil(duration_seconds / seconds_per_bar)), seconds_per_bar

    def create_backing_track(self, num_bars=8, tempo=120):
        """Create a synthwave-style backing track with drums, bass, and synth chords."""
        beats_per_bar = 4
        seconds_per_bar = 60.0 / tempo * beats_per_bar
        print(f"[DEBUG] Backing: tempo={tempo}, beats_per_bar={beats_per_bar}, seconds_per_bar={seconds_per_bar}")
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        chord_progression = [
            [60, 64, 67],
            [67, 71, 74],
            [57, 60, 64],
            [65, 69, 72],
        ]
        full_progression = (chord_progression * ((num_bars // len(chord_progression)) + 1))[:num_bars]
        synth_pad = pretty_midi.Instrument(program=89)
        note_debug_count = 0
        for i, chord in enumerate(full_progression):
            start_time = i * seconds_per_bar
            end_time = start_time + seconds_per_bar
            for note_pitch in chord:
                note = pretty_midi.Note(velocity=70, pitch=note_pitch, start=start_time, end=end_time)
                synth_pad.notes.append(note)
                note = pretty_midi.Note(velocity=65, pitch=note_pitch + 12, start=start_time, end=end_time)
                synth_pad.notes.append(note)
                if note_debug_count < 3:
                    print(f"[DEBUG] Backing: Note pitch={note_pitch}, start={start_time}, end={end_time}")
                    note_debug_count += 1
        midi.instruments.append(synth_pad)
        synth_bass = pretty_midi.Instrument(program=87)
        for i, chord in enumerate(full_progression):
            start_time = i * seconds_per_bar
            sixteenth_note_duration = seconds_per_bar / 16
            for j in range(16):
                note_start = start_time + (j * sixteenth_note_duration)
                note_end = note_start + (sixteenth_note_duration / 4)
                beat_position = j % 4
                if beat_position == 0 and j > 0:
                    continue
                chord_position = j % len(chord)
                pitch = chord[chord_position] - 24
                velocity = 100 if j % 4 == 0 else 90
                note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=note_start, end=note_end)
                synth_bass.notes.append(note)
        midi.instruments.append(synth_bass)
        drums = pretty_midi.Instrument(program=0, is_drum=True)
        for measure in range(num_bars):
            measure_start = measure * seconds_per_bar
            for beat in range(4):
                beat_time = measure_start + (beat * seconds_per_bar / 4)
                kick = pretty_midi.Note(velocity=100, pitch=36, start=beat_time, end=beat_time + 0.05)
                drums.notes.append(kick)
                if beat in [1, 3]:
                    snare = pretty_midi.Note(velocity=90, pitch=38, start=beat_time, end=beat_time + 0.05)
                    drums.notes.append(snare)
                for j in range(4):
                    hihat_time = beat_time + (j * seconds_per_bar / 16)
                    velocity = 75 if j in [0, 2] else 70
                    hihat = pretty_midi.Note(velocity=velocity, pitch=42, start=hihat_time, end=hihat_time + 0.025)
                    drums.notes.append(hihat)
        midi.instruments.append(drums)
        actual_tempos = midi.get_tempo_changes()[1]
        print(f"[DEBUG] Backing track: actual MIDI tempos: {actual_tempos}")
        return midi

    def create_seed_melody(self, key=60, num_bars=8, tempo=120):
        """Create a seed melody for generating a lead melody."""
        beats_per_bar = 4
        seconds_per_bar = 60.0 / tempo * beats_per_bar
        print(f"[DEBUG] Seed: tempo={tempo}, beats_per_bar={beats_per_bar}, seconds_per_bar={seconds_per_bar}")
        seed_midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        synth_lead = pretty_midi.Instrument(program=81)
        base_key = key + 12
        scale = [base_key, base_key + 2, base_key + 4, base_key + 5, base_key + 7, base_key + 9, base_key + 11]
        quarter_note = seconds_per_bar / 4
        eighth_note = seconds_per_bar / 8
        sixteenth_note = seconds_per_bar / 16
        # Use more natural synthwave/guitar rhythms
        pattern1 = [
            (scale[0], quarter_note),
            (scale[4], eighth_note),
            (scale[2], eighth_note),
            (scale[4], quarter_note),
        ]
        pattern2 = [
            (scale[4], eighth_note),
            (scale[0], eighth_note),
            (scale[5], quarter_note),
            (scale[2], quarter_note),
        ]
        pattern3 = []
        for i in range(4):
            pattern3.append((scale[i % 7], sixteenth_note))
            pattern3.append((scale[(i + 2) % 7], eighth_note))
        pattern4 = []
        for i in range(4):
            pattern4.append((scale[6 - (i % 7)], sixteenth_note))
            pattern4.append((scale[6 - (i % 7)], eighth_note))
        all_patterns = pattern1 + pattern2 + pattern3 + pattern4
        bars_per_pattern = 1
        total_patterns = int(np.ceil(num_bars / bars_per_pattern))
        repeated_patterns = (all_patterns * ((total_patterns * 8) // len(all_patterns) + 1))[:num_bars * 8]
        current_time = 0
        note_debug_count = 0
        for pitch, duration in repeated_patterns:
            if current_time + duration > num_bars * seconds_per_bar:
                break
            note = pretty_midi.Note(velocity=90, pitch=pitch, start=current_time, end=current_time + duration)
            synth_lead.notes.append(note)
            if note_debug_count < 3:
                print(f"[DEBUG] Seed: Note pitch={pitch}, start={current_time}, end={current_time + duration}")
                note_debug_count += 1
            current_time += duration
        seed_midi.instruments.append(synth_lead)
        actual_tempos = seed_midi.get_tempo_changes()[1]
        print(f"[DEBUG] Seed melody: actual MIDI tempos: {actual_tempos}")
        return seed_midi

    def generate_lead_melody(self, backing_midi, num_bars=8, tempo=120):
        """Generate a lead melody using AMT over the backing track and a seed melody."""
        print("Converting backing track to events...")
        beats_per_bar = 4
        seconds_per_bar = 60.0 / tempo * beats_per_bar
        seed_midi = self.create_seed_melody(key=60, num_bars=num_bars, tempo=tempo)
        
        # Save the seed melody for reference
        seed_path = self.output_dir / 'seed_melody.mid'
        seed_midi.write(str(seed_path))
        print(f"Saved seed melody to {seed_path}")

        temp_seed_file = self.output_dir / 'temp_seed.mid'
        seed_midi.write(str(temp_seed_file))
        seed_events = midi_to_events(str(temp_seed_file))
        os.remove(temp_seed_file)

        # Convert backing track to events
        temp_backing_file = self.output_dir / 'temp_backing.mid'
        backing_midi.write(str(temp_backing_file))
        backing_events = midi_to_events(str(temp_backing_file))
        os.remove(temp_backing_file)

        # Use a subset of seed events for conditioning (every other event)
        seed_subset = []
        for event in seed_events:
            if len(seed_subset) % 2 == 0:
                seed_subset.append(event)

        # Combine subset of seed events with backing events
        combined_events = backing_events + seed_subset

        print(f"Generating lead melody with {len(combined_events)} conditioning events...")

        try:
            # Generate multiple variations and select the most interesting one
            num_variations = 3
            best_melody = None
            best_note_count = 0

            for i in range(num_variations):
                # Generate with combined seed conditioning
                generated_events = generate(
                    self.model,
                    start_time=0,
                    end_time=num_bars * seconds_per_bar,
                    controls=combined_events,
                    top_p=0.9  # Slightly higher for more creative generation
                )

                # Convert back to MIDI
                generated_midi = events_to_midi(generated_events)

                # Convert to pretty_midi for easier manipulation
                temp_file = self.output_dir / f'temp_generated_{i}.mid'
                generated_midi.save(str(temp_file))
                pretty_generated = pretty_midi.PrettyMIDI(str(temp_file))
                os.remove(temp_file)

                # Process the generated melody
                valid_notes = []
                current_time = 0

                # Sort all notes by start time
                all_notes = []
                for instrument in pretty_generated.instruments:
                    if not instrument.is_drum:
                        all_notes.extend(instrument.notes)
                all_notes.sort(key=lambda x: x.start)

                # Filter for monophonic melody with more variety
                for note in all_notes:
                    # Keep notes in guitar-like range (72-96)
                    if 72 <= note.pitch <= 96:
                        # Ensure valid timing
                        start_time = max(0, note.start)
                        end_time = min(num_bars * seconds_per_bar, note.end)

                        # Skip notes with invalid timing
                        if end_time <= start_time:
                            continue

                        # Skip if this note overlaps with the current note
                        if start_time < current_time:
                            continue

                        # Add more variation to note durations
                        duration = min(0.5, max(0.0625, end_time - start_time))

                        # Adjust velocity for better dynamics
                        base_velocity = min(120, max(90, note.velocity))
                        velocity = int(base_velocity * (0.9 + (0.2 * (note.pitch % 12) / 12)))

                        # Create a new note with adjusted parameters
                        new_note = pretty_midi.Note(
                            velocity=velocity,
                            pitch=note.pitch,
                            start=start_time,
                            end=start_time + duration
                        )
                        valid_notes.append(new_note)
                        current_time = new_note.end

                # Select the variation with the most notes (most interesting)
                if len(valid_notes) > best_note_count:
                    best_note_count = len(valid_notes)
                    best_melody = valid_notes

            if best_melody:
                # Create a new MIDI file for the melody
                melody_midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
                synth_lead = pretty_midi.Instrument(program=81)  # Lead 2 (sawtooth)
                synth_lead.notes = best_melody
                melody_midi.instruments.append(synth_lead)
                return melody_midi
            else:
                print("No suitable notes found in any generated variation")
                return None

        except Exception as e:
            print(f"Generation failed: {e}")
            return None

    def preview_audio(self, midi_file, sample_rate=44100):
        """Convert MIDI to audio and return an IPython Audio object for preview."""
        audio_data = self.midi_to_audio(midi_file, sample_rate)
        return Audio(audio_data, rate=sample_rate)

    def save_and_preview(self, midi_file, filename):
        """Save MIDI file and return audio preview."""
        output_path = self.output_dir / filename
        midi_file.write(str(output_path))
        print(f"Saved to {output_path}")
        
        # Also save the audio file if we're using a soundfont
        if self.soundfont_path is not None:
            audio_path = output_path.with_suffix('.wav')
            audio_data = self.midi_to_audio(midi_file)
            # Convert to 16-bit PCM
            audio_data = (audio_data * 32768).astype(np.int16)
            with open(audio_path, 'wb') as f:
                f.write(audio_data.tobytes())
            print(f"Saved audio to {audio_path}")
        
        return self.preview_audio(midi_file)

    def _parse_chord(self, chord_name: str, key_offset: int = 0) -> list[int]:
        """Parse a chord name (e.g., 'C', 'Am', 'F#m7') and return MIDI note numbers."""
        # Define note names and their semitone offsets from C
        note_names = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
            'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
        }
        
        # Define chord types and their intervals
        chord_types = {
            '': [0, 4, 7],  # Major
            'm': [0, 3, 7],  # Minor
            'dim': [0, 3, 6],  # Diminished
            'aug': [0, 4, 8],  # Augmented
            '7': [0, 4, 7, 10],  # Dominant 7th
            'm7': [0, 3, 7, 10],  # Minor 7th
            'maj7': [0, 4, 7, 11],  # Major 7th
            'dim7': [0, 3, 6, 9],  # Diminished 7th
            'sus2': [0, 2, 7],  # Suspended 2nd
            'sus4': [0, 5, 7],  # Suspended 4th
        }
        
        # Parse the chord name
        chord_name = chord_name.strip()
        
        # Extract the root note and chord type
        # Handle sharp/flat notes
        if len(chord_name) >= 2 and chord_name[1] in '#b':
            root_note = chord_name[:2]
            chord_type = chord_name[2:]
        else:
            root_note = chord_name[0]
            chord_type = chord_name[1:]
        
        # Get the root note offset
        if root_note not in note_names:
            raise ValueError(f"Unknown note name: {root_note}")
        
        root_offset = note_names[root_note]
        
        # Get the chord intervals
        if chord_type not in chord_types:
            # Default to major if unknown
            intervals = chord_types['']
        else:
            intervals = chord_types[chord_type]
        
        # Calculate MIDI note numbers (C4 = 60)
        base_note = 60 + key_offset + root_offset
        chord_notes = [base_note + interval for interval in intervals]
        
        return chord_notes

    def _get_key_offset(self, key: str) -> int:
        """Convert a key name to its semitone offset from C."""
        key_offsets = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
            'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
        }
        
        if key not in key_offsets:
            raise ValueError(f"Unknown key: {key}")
        
        return key_offsets[key]

    def create_backing_track_with_chords(self, chord_progression: list[str], key: str = "C", num_bars: int = 8, tempo: int = 120):
        """Create a backing track with custom chord progression including drums, bass, and piano."""
        beats_per_bar = 4
        seconds_per_bar = 60.0 / tempo * beats_per_bar
        print(f"[DEBUG] Backing with chords: tempo={tempo}, key={key}, chords={chord_progression}")
        
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        
        # Get key offset
        key_offset = self._get_key_offset(key)
        
        # Parse chords to get note numbers
        chord_notes_list = []
        for chord in chord_progression:
            try:
                notes = self._parse_chord(chord, key_offset)
                chord_notes_list.append(notes)
                print(f"[DEBUG] Chord {chord} -> notes {notes}")
            except ValueError as e:
                print(f"Error parsing chord {chord}: {e}")
                # Fallback to C major
                chord_notes_list.append([60, 64, 67])
        
        # Repeat chord progression to fill num_bars
        full_progression = (chord_notes_list * ((num_bars // len(chord_notes_list)) + 1))[:num_bars]
        
        # Piano/Chord part
        piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
        for i, chord in enumerate(full_progression):
            start_time = i * seconds_per_bar
            end_time = start_time + seconds_per_bar
            for note_pitch in chord:
                # Add root note and octave
                note = pretty_midi.Note(velocity=80, pitch=note_pitch, start=start_time, end=end_time)
                piano.notes.append(note)
                # Add octave above for fuller sound
                note = pretty_midi.Note(velocity=70, pitch=note_pitch + 12, start=start_time, end=end_time)
                piano.notes.append(note)
        midi.instruments.append(piano)
        
        # Bass part
        bass = pretty_midi.Instrument(program=32)  # Acoustic Bass
        for i, chord in enumerate(full_progression):
            start_time = i * seconds_per_bar
            sixteenth_note_duration = seconds_per_bar / 16
            for j in range(16):
                note_start = start_time + (j * sixteenth_note_duration)
                note_end = note_start + (sixteenth_note_duration / 4)
                beat_position = j % 4
                if beat_position == 0 and j > 0:
                    # Play root note of chord on beat
                    root_note = chord[0] - 12  # One octave below
                    note = pretty_midi.Note(velocity=90, pitch=root_note, start=note_start, end=note_end)
                    bass.notes.append(note)
                elif beat_position == 2:
                    # Play fifth on off-beat
                    if len(chord) >= 3:
                        fifth_note = chord[2] - 12  # One octave below
                        note = pretty_midi.Note(velocity=70, pitch=fifth_note, start=note_start, end=note_end)
                        bass.notes.append(note)
        midi.instruments.append(bass)
        
        # Drums part
        drums = pretty_midi.Instrument(program=0, is_drum=True)
        for i in range(num_bars):
            start_time = i * seconds_per_bar
            beat_duration = seconds_per_bar / 4
            
            # Kick drum on beats 1 and 3
            kick_note = pretty_midi.Note(velocity=100, pitch=36, start=start_time, end=start_time + 0.1)
            drums.notes.append(kick_note)
            kick_note = pretty_midi.Note(velocity=100, pitch=36, start=start_time + 2*beat_duration, end=start_time + 2*beat_duration + 0.1)
            drums.notes.append(kick_note)
            
            # Snare on beats 2 and 4
            snare_note = pretty_midi.Note(velocity=90, pitch=38, start=start_time + beat_duration, end=start_time + beat_duration + 0.1)
            drums.notes.append(snare_note)
            snare_note = pretty_midi.Note(velocity=90, pitch=38, start=start_time + 3*beat_duration, end=start_time + 3*beat_duration + 0.1)
            drums.notes.append(snare_note)
            
            # Hi-hat on every eighth note
            eighth_duration = seconds_per_bar / 8
            for j in range(8):
                hat_start = start_time + j * eighth_duration
                hat_note = pretty_midi.Note(velocity=60, pitch=42, start=hat_start, end=hat_start + 0.05)
                drums.notes.append(hat_note)
        
        midi.instruments.append(drums)
        return midi 