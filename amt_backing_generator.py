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
            
            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if result.returncode != 0:
                print(f"Fluidsynth error output: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
            
            # Read the generated audio file
            audio_data = np.frombuffer(open(temp_audio_path, 'rb').read(), dtype=np.int16)
            # Convert to float32 and normalize
            audio_data = audio_data.astype(np.float32) / 32768.0
            
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
            except:
                pass

    def create_backing_track(self, duration_seconds=24, tempo=240):
        """Create a synthwave-style backing track with drums, bass, and synth chords."""
        # Convert BPM to microseconds per quarter note
        microseconds_per_quarter = int(60000000 / tempo)
        
        # Create MIDI file with the correct tempo
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        
        # Add synth pad for chords (using a warm pad sound)
        synth_pad = pretty_midi.Instrument(program=89)  # Warm Pad
        
        # Synthwave chord progression in C major: C - G - Am - F
        # Extended to 8 bars by repeating the progression
        chord_progression = [
            [60, 64, 67],  # C major (C4, E4, G4)
            [67, 71, 74],  # G major (G4, B4, D5)
            [57, 60, 64],  # A minor (A3, C4, E4)
            [65, 69, 72],  # F major (F4, A4, C5)
        ] * 2  # Repeat the progression for 8 bars
        
        # Calculate actual time per bar at 240 BPM
        seconds_per_bar = 60.0 / tempo * 4  # 4 beats per bar
        
        # Add sustained chords with synth pad
        for i, chord in enumerate(chord_progression):
            start_time = i * seconds_per_bar
            end_time = start_time + seconds_per_bar
            
            # Add the main chord notes
            for note_pitch in chord:
                # Main chord note
                note = pretty_midi.Note(
                    velocity=70,  # Moderate velocity for pad
                    pitch=note_pitch,
                    start=start_time,
                    end=end_time
                )
                synth_pad.notes.append(note)
                
                # Add an octave up for more fullness
                note = pretty_midi.Note(
                    velocity=65,
                    pitch=note_pitch + 12,
                    start=start_time,
                    end=end_time
                )
                synth_pad.notes.append(note)
        
        midi.instruments.append(synth_pad)
        
        # Add synthwave bass with steady 16th notes
        synth_bass = pretty_midi.Instrument(program=87)  # Lead 8 (bass + lead)
        
        for i, chord in enumerate(chord_progression):
            start_time = i * seconds_per_bar
            sixteenth_note_duration = seconds_per_bar / 16  # Duration of a 16th note
            
            # Create a steady 16th note pattern
            for j in range(16):  # 16 16th notes per bar
                note_start = start_time + (j * sixteenth_note_duration)
                note_end = note_start + (sixteenth_note_duration / 4)  # Very short notes for tight bass
                
                # Skip the first 16th note of beats 2, 3, and 4
                beat_position = j % 4
                if beat_position == 0 and j > 0:  # If it's the first 16th note of beats 2, 3, or 4
                    continue
                
                # Cycle through chord notes for arpeggiation
                chord_position = j % len(chord)
                pitch = chord[chord_position] - 24  # Lower the bass by two octaves for better separation
                
                # Adjust velocity based on position in bar
                if j % 4 == 0:  # On the beat
                    velocity = 100
                else:
                    velocity = 90
                
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=note_start,
                    end=note_end
                )
                synth_bass.notes.append(note)
        
        midi.instruments.append(synth_bass)
        
        # Add drum track with steady 16th note hi-hats
        drums = pretty_midi.Instrument(program=0, is_drum=True)
        measures = 8  # Extended to 8 measures
        
        for measure in range(measures):
            measure_start = measure * seconds_per_bar
            
            for beat in range(4):  # 4 beats per bar
                beat_time = measure_start + (beat * seconds_per_bar / 4)
                
                # Kick drum on all beats (1, 2, 3, 4)
                kick = pretty_midi.Note(velocity=100, pitch=36, start=beat_time, end=beat_time + 0.05)
                drums.notes.append(kick)
                
                # Snare on beats 2 and 4
                if beat in [1, 3]:
                    snare = pretty_midi.Note(velocity=90, pitch=38, start=beat_time, end=beat_time + 0.05)
                    drums.notes.append(snare)
                
                # Steady 16th note hi-hat pattern
                for j in range(4):  # 4 16th notes per beat
                    hihat_time = beat_time + (j * seconds_per_bar / 16)
                    velocity = 75 if j in [0, 2] else 70
                    hihat = pretty_midi.Note(velocity=velocity, pitch=42, start=hihat_time, end=hihat_time + 0.025)
                    drums.notes.append(hihat)
        
        midi.instruments.append(drums)
        return midi

    def create_seed_melody(self, key=60, duration_seconds=24):
        """Create a seed melody using synthwave-style patterns in the given key."""
        # Create a MIDI file for the seed melody
        seed_midi = pretty_midi.PrettyMIDI(initial_tempo=240)
        
        # Create a synth lead instrument
        synth_lead = pretty_midi.Instrument(program=81)  # Lead 2 (sawtooth)
        
        # Define a major scale with some additional notes for more interest
        # Using C major with added notes (C, D, E, F, G, A, B)
        # Keep everything in a reasonable guitar range (72-96)
        base_key = 72  # Start at C5 (72)
        scale = [base_key, base_key + 2, base_key + 4, base_key + 5, base_key + 7, base_key + 9, base_key + 11]
        
        # Calculate timing based on tempo
        seconds_per_bar = 60.0 / 240 * 4  # 4 beats per bar at 240 BPM
        sixteenth_note = seconds_per_bar / 4  # For consistent rhythm
        
        # Pattern 1: Syncopated rhythm with chord tones (bars 1-2)
        pattern1 = [
            (scale[0], sixteenth_note * 1.5),      # Held note
            (scale[4], sixteenth_note * 0.5),      # Quick note
            (scale[2], sixteenth_note),            # Regular note
            (scale[4], sixteenth_note),            # Regular note
            (scale[6], sixteenth_note * 1.5),      # Held note
            (scale[4], sixteenth_note * 0.5),      # Quick note
            (scale[2], sixteenth_note),            # Regular note
            (scale[0], sixteenth_note),            # Regular note
        ]
        
        # Pattern 2: Alternating high and low notes (bars 3-4)
        pattern2 = [
            (scale[4], sixteenth_note),            # High note
            (scale[0], sixteenth_note),            # Low note
            (scale[5], sixteenth_note),            # High note
            (scale[2], sixteenth_note),            # Low note
            (scale[6], sixteenth_note),            # High note
            (scale[4], sixteenth_note),            # Low note
            (scale[0], sixteenth_note),            # High note
            (scale[5], sixteenth_note),            # Low note
        ]
        
        # Pattern 3: Ascending sequence with rhythm (bars 5-6)
        pattern3 = []
        for i in range(8):
            if i % 2 == 0:
                pattern3.append((scale[i % 7], sixteenth_note * 1.5))  # Held note
            else:
                pattern3.append((scale[(i + 2) % 7], sixteenth_note * 0.5))  # Quick note
        
        # Pattern 4: Descending sequence with rhythm (bars 7-8)
        pattern4 = []
        for i in range(8):
            if i % 2 == 0:
                pattern4.append((scale[6 - (i % 7)], sixteenth_note))  # Regular note
            else:
                pattern4.append((scale[6 - (i % 7)], sixteenth_note))  # Regular note
        
        # Combine all patterns
        all_patterns = pattern1 + pattern2 + pattern3 + pattern4
        
        # Add notes to the track
        current_time = 0
        for pitch, duration in all_patterns:
            note = pretty_midi.Note(
                velocity=90,
                pitch=pitch,
                start=current_time,
                end=current_time + duration
            )
            synth_lead.notes.append(note)
            current_time += duration
        
        seed_midi.instruments.append(synth_lead)
        return seed_midi

    def generate_lead_melody(self, backing_midi, duration_seconds=24):
        """Generate a lead melody using AMT over the backing track and a seed melody."""
        print("Converting backing track to events...")

        # Create and convert seed melody to events
        print("Creating seed melody for better generation...")
        seed_midi = self.create_seed_melody(key=60)  # C major

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
                    end_time=duration_seconds,
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
                        end_time = min(duration_seconds, note.end)

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
                melody_midi = pretty_midi.PrettyMIDI(initial_tempo=240)
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

def main():
    # Initialize generator with soundfont
    # You can specify your own soundfont path here
    generator = AMTBackingGenerator(soundfont_path="/path/to/your/soundfont.sf2")
    
    # Generate backing track
    print("\nGenerating backing track...")
    backing_track = generator.create_backing_track()
    print("Backing track generated. Preview:")
    backing_audio = generator.save_and_preview(backing_track, "backing_track.mid")
    
    # Generate lead melody
    print("\nGenerating lead melody...")
    lead_melody = generator.generate_lead_melody(backing_track)
    if lead_melody:
        print("Lead melody generated. Preview:")
        lead_audio = generator.save_and_preview(lead_melody, "lead_melody.mid")
        
        # Combine tracks
        combined = pretty_midi.PrettyMIDI()
        for instrument in backing_track.instruments:
            combined.instruments.append(instrument)
        for instrument in lead_melody.instruments:
            combined.instruments.append(instrument)
        
        print("\nCombined track preview:")
        combined_audio = generator.save_and_preview(combined, "combined_track.mid")
    else:
        print("Failed to generate lead melody")

if __name__ == "__main__":
    main() 