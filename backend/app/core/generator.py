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
import os
from pathlib import Path
import subprocess
import tempfile
import re

class AMTBackingGenerator:
    def __init__(self, model_name='stanford-crfm/music-medium-800k', soundfont_path=None):
        """Initialize the AMT backing generator with the specified model and soundfont."""
        print("Loading AMT model...")
        print(f"[DEBUG] Using model: {model_name}")
        
        # Validate model name
        if 'music-medium' not in model_name and 'music-large' not in model_name:
            print(f"[DEBUG] Warning: Model name '{model_name}' may not be compatible with anticipation library")
            print("[DEBUG] Expected models: stanford-crfm/music-medium-800k or stanford-crfm/music-large-800k")
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model = self.model.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
        print(f"Model config: {self.model.config}")
        
        # Check model configuration for anticipation compatibility
        if hasattr(self.model.config, 'vocab_size'):
            print(f"[DEBUG] Model vocab size: {self.model.config.vocab_size}")
        if hasattr(self.model.config, 'max_position_embeddings'):
            print(f"[DEBUG] Model max position embeddings: {self.model.config.max_position_embeddings}")
        
        # Check if model is properly configured for anticipation
        try:
            # Test if the model can handle basic generation
            test_input = torch.tensor([[0]], device=self.device, dtype=torch.long)
            with torch.no_grad():
                test_output = self.model(test_input)
            print("[DEBUG] Model test generation successful")
        except Exception as e:
            print(f"[DEBUG] Warning: Model test generation failed: {e}")
            print("[DEBUG] This might indicate compatibility issues with the anticipation library")
        
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
        """Create a seed melody for just the first measure using random eighth notes."""
        print("[DEBUG] Creating seed melody")
        beats_per_bar = 4
        seconds_per_bar = 60.0 / tempo * beats_per_bar
        print(f"[DEBUG] Seed: tempo={tempo}, beats_per_bar={beats_per_bar}, seconds_per_bar={seconds_per_bar}")

        # Create a MIDI file
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        synth_lead = pretty_midi.Instrument(program=81)  # Lead 2 (sawtooth)

        # Define the scale (C major scale starting from the root note)
        scale = []
        root_note = key  # Use the provided key as the root note
        # Add one octave of notes in the major scale
        for i in range(8):  # 8 notes in an octave
            if i == 0:  # Root
                scale.append(root_note)
            elif i == 1:  # Major second
                scale.append(root_note + 2)
            elif i == 2:  # Major third
                scale.append(root_note + 4)
            elif i == 3:  # Perfect fourth
                scale.append(root_note + 5)
            elif i == 4:  # Perfect fifth
                scale.append(root_note + 7)
            elif i == 5:  # Major sixth
                scale.append(root_note + 9)
            elif i == 6:  # Major seventh
                scale.append(root_note + 11)
            elif i == 7:  # Octave
                scale.append(root_note + 12)
        
        print(f"[DEBUG] Generated scale with {len(scale)} notes: {scale}")
        
        # Calculate eighth note duration
        eighth_note_duration = seconds_per_bar / 8  # 8 eighth notes per bar
        
        # Generate random eighth notes for just the first measure
        current_time = 0
        for _ in range(8):  # 8 eighth notes in a measure
            # Randomly select a note from the scale
            scale_index = np.random.randint(0, len(scale))  # Use len(scale) to ensure valid index
            pitch = scale[scale_index]
            
            # Random velocity between 85 and 105 for natural dynamics
            velocity = np.random.randint(85, 106)
            
            # Create the note
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=current_time,
                end=current_time + eighth_note_duration
            )
            synth_lead.notes.append(note)
            print(f"[DEBUG] Seed: Note pitch={pitch}, start={current_time}, end={current_time + eighth_note_duration}, velocity={velocity}")
            
            # Move to next eighth note
            current_time += eighth_note_duration

        midi.instruments.append(synth_lead)
        return midi

    def generate_lead_melody(self, backing_midi, num_bars=8, tempo=120):
        """Generate a lead melody using AMT over the backing track and a seed melody."""
        print("[DEBUG] Starting lead melody generation")
        beats_per_bar = 4
        seconds_per_bar = 60.0 / tempo * beats_per_bar
        print(f"[DEBUG] Lead melody params: num_bars={num_bars}, tempo={tempo}, seconds_per_bar={seconds_per_bar}")
        
        # Create seed melody for just the first bar (4 beats)
        print("[DEBUG] Creating seed melody for first bar only")
        seed_midi = self.create_seed_melody(key=60, num_bars=1, tempo=tempo)  # Only 1 bar as seed
        print(f"[DEBUG] Seed melody created with {len(seed_midi.instruments)} instruments")
        if len(seed_midi.instruments) > 0:
            print(f"[DEBUG] First instrument has {len(seed_midi.instruments[0].notes)} notes")
            for i, note in enumerate(seed_midi.instruments[0].notes):
                print(f"[DEBUG] Seed note {i}: pitch={note.pitch}, start={note.start}, end={note.end}, velocity={note.velocity}")
        
        # Save the seed melody for reference
        seed_path = self.output_dir / 'seed_melody.mid'
        try:
            seed_midi.write(str(seed_path))
            print(f"[DEBUG] Saved seed melody to {seed_path}")
        except Exception as e:
            print(f"[DEBUG] Error saving seed melody: {e}")
            return None

        # Convert seed melody to events
        print("[DEBUG] Converting seed melody to events")
        temp_seed_file = self.output_dir / 'temp_seed.mid'
        try:
            seed_midi.write(str(temp_seed_file))
            print(f"[DEBUG] Wrote temporary seed file to {temp_seed_file}")
            
            # Verify the file exists and has content
            if not temp_seed_file.exists():
                print("[DEBUG] Error: Temporary seed file was not created")
                return None
            if temp_seed_file.stat().st_size == 0:
                print("[DEBUG] Error: Temporary seed file is empty")
                return None
                
            print("[DEBUG] Attempting to convert seed MIDI to events")
            seed_events = midi_to_events(str(temp_seed_file))
            print(f"[DEBUG] Successfully converted seed melody to {len(seed_events)} events")
            
            if len(seed_events) > 0:
                print(f"[DEBUG] First seed event: {seed_events[0]}")
                print(f"[DEBUG] Last seed event: {seed_events[-1]}")
                print(f"[DEBUG] Seed event types: {[type(event) for event in seed_events[:5]]}")
            else:
                print("[DEBUG] Warning: No seed events were generated")
                
        except Exception as e:
            print(f"[DEBUG] Error converting seed melody to events: {e}")
            import traceback
            print(f"[DEBUG] Seed conversion traceback: {traceback.format_exc()}")
            seed_events = []
        finally:
            try:
                os.remove(temp_seed_file)
                print("[DEBUG] Cleaned up temporary seed file")
            except Exception as e:
                print(f"[DEBUG] Warning: Could not remove temporary seed file: {e}")

        if not seed_events:
            print("[DEBUG] Error: No valid seed events available for generation")
            return None

        # Validate seed events
        print("[DEBUG] Validating seed events")
        valid_seed_events = []
        for i, event in enumerate(seed_events):
            if isinstance(event, int) and event >= 0:
                valid_seed_events.append(event)
            else:
                print(f"[DEBUG] Invalid seed event {i}: {event} (type: {type(event)})")
        
        if not valid_seed_events:
            print("[DEBUG] Error: No valid seed events after validation")
            return None
        
        print(f"[DEBUG] Using {len(valid_seed_events)} valid seed events for generation")
        seed_events = valid_seed_events

        print(f"[DEBUG] Starting AMT generation with {len(seed_events)} seed events")
        
        try:
            # Add more debugging for the model state
            print(f"[DEBUG] Model device: {self.model.device}")
            print(f"[DEBUG] Model dtype: {next(self.model.parameters()).dtype}")
            print(f"[DEBUG] Generation start: {0}, end: {num_bars * seconds_per_bar}")
            print(f"[DEBUG] Number of seed events: {len(seed_events)}")
            print(f"[DEBUG] Seed events: {seed_events[:10]}...")  # Show first 10 events
            
            # Try a different approach: generate full 8 bars with strong seed conditioning
            # The issue might be that the model doesn't understand partial generation well
            try:
                print("[DEBUG] Attempting full 8-bar generation with seed conditioning")
                generated_events = generate(
                    self.model,
                    start_time=0,  # Start from beginning
                    end_time=num_bars * seconds_per_bar,  # Full 8 bars
                    controls=seed_events,  # Use seed events as conditioning
                    top_p=0.7  # More conservative for better conditioning
                )
            except Exception as amt_error:
                print(f"[DEBUG] AMT generate function failed: {amt_error}")
                print("[DEBUG] This might be a compatibility issue with the anticipation library")
                print("[DEBUG] Attempting alternative generation approach...")
                
                # Try with different parameters
                try:
                    generated_events = generate(
                        self.model,
                        start_time=0,
                        end_time=num_bars * seconds_per_bar,
                        controls=seed_events,
                        top_p=0.9
                    )
                except Exception as alt_error:
                    print(f"[DEBUG] Alternative generation also failed: {alt_error}")
                    print("[DEBUG] Skipping this variation due to AMT compatibility issues")
                    return None
            
            print(f"[DEBUG] Generated {len(generated_events)} events for variation")
            if len(generated_events) > 0:
                print(f"[DEBUG] First generated event: {generated_events[0]}")
                print(f"[DEBUG] Last generated event: {generated_events[-1]}")
                print(f"[DEBUG] Event types: {[type(event) for event in generated_events[:5]]}")
                print(f"[DEBUG] First 5 events: {generated_events[:5]}")
                
                # AMT events should be non-negative integers
                if len(generated_events) > 0:
                    min_event = min(generated_events)
                    max_event = max(generated_events)
                    print(f"[DEBUG] Event range: min={min_event}, max={max_event}")
                    
                    # Check for any non-integer events
                    non_integer_events = [event for event in generated_events if not isinstance(event, int)]
                    if non_integer_events:
                        print(f"[DEBUG] Found {len(non_integer_events)} non-integer events: {non_integer_events[:5]}")
                    
                    # Filter out negative events (these are invalid)
                    filtered_events = [event for event in generated_events if event >= 0]
                    print(f"[DEBUG] Filtered from {len(generated_events)} to {len(filtered_events)} non-negative events")
                    
                    if len(filtered_events) == 0:
                        print("[DEBUG] No valid events after filtering, skipping this variation")
                        return None
                    
                    generated_events = filtered_events
                
        except Exception as e:
            print(f"[DEBUG] Error in AMT generation: {e}")
            import traceback
            print(f"[DEBUG] AMT generation traceback: {traceback.format_exc()}")
            return None

        # Convert back to MIDI
        try:
            print(f"[DEBUG] Converting variation to MIDI with {len(generated_events)} events")
            generated_midi = events_to_midi(generated_events)
            print(f"[DEBUG] Successfully converted variation to MIDI")
            
            # Debug: Let's see what we got (skip the instruments check since it's a MidiFile, not PrettyMIDI)
            print(f"[DEBUG] Generated MIDI file created successfully")
            
        except Exception as e:
            print(f"[DEBUG] Error converting variation to MIDI: {e}")
            import traceback
            print(f"[DEBUG] MIDI conversion traceback: {traceback.format_exc()}")
            return None

        # Convert to pretty_midi for easier manipulation
        temp_file = self.output_dir / f'temp_generated.mid'
        try:
            print(f"[DEBUG] Saving variation to temporary file")
            generated_midi.save(str(temp_file))
            print(f"[DEBUG] Loading variation into pretty_midi")
            pretty_generated = pretty_midi.PrettyMIDI(str(temp_file))
            print(f"[DEBUG] Successfully loaded variation into pretty_midi")
        except Exception as e:
            print(f"[DEBUG] Error processing variation MIDI: {e}")
            import traceback
            print(f"[DEBUG] MIDI processing traceback: {traceback.format_exc()}")
            return None
        finally:
            try:
                os.remove(temp_file)
                print(f"[DEBUG] Cleaned up temporary file for variation")
            except Exception as e:
                print(f"[DEBUG] Warning: Could not remove temporary file for variation: {e}")

        # Process the generated melody
        try:
            print(f"[DEBUG] Processing notes for variation")
            valid_notes = []
            current_time = 0

            # Sort all notes by start time
            all_notes = []
            for instrument in pretty_generated.instruments:
                if not instrument.is_drum:
                    all_notes.extend(instrument.notes)
            all_notes.sort(key=lambda x: x.start)
            print(f"[DEBUG] Found {len(all_notes)} notes in variation")

            # Filter for monophonic melody with more variety
            for note in all_notes:
                # Keep notes in guitar-like range (60-84 for better range)
                if 60 <= note.pitch <= 84:
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

            print(f"[DEBUG] Filtered to {len(valid_notes)} valid notes in variation")
            if len(valid_notes) > 0:
                print(f"[DEBUG] First valid note: pitch={valid_notes[0].pitch}, start={valid_notes[0].start}, end={valid_notes[0].end}")
                print(f"[DEBUG] Last valid note: pitch={valid_notes[-1].pitch}, start={valid_notes[-1].start}, end={valid_notes[-1].end}")

            best_melody = valid_notes
            print(f"[DEBUG] New best variation found with {len(best_melody)} notes")

        except Exception as e:
            print(f"[DEBUG] Error processing notes for variation: {e}")
            import traceback
            print(f"[DEBUG] Note processing traceback: {traceback.format_exc()}")
            return None

        if best_melody:
            print(f"[DEBUG] Creating final melody with {len(best_melody)} notes")
            try:
                # Create a new MIDI file for the melody
                melody_midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
                synth_lead = pretty_midi.Instrument(program=81)  # Lead 2 (sawtooth)
                
                # Add the seed melody notes first (bar 1)
                seed_note_count = 0
                for note in seed_midi.instruments[0].notes:
                    synth_lead.notes.append(note)
                    seed_note_count += 1
                print(f"[DEBUG] Added {seed_note_count} seed notes to final melody")
                
                # Add the generated notes (bars 2-8)
                generated_note_count = 0
                for note in best_melody:
                    synth_lead.notes.append(note)
                    generated_note_count += 1
                print(f"[DEBUG] Added {generated_note_count} generated notes to final melody")
                
                melody_midi.instruments.append(synth_lead)
                
                # Debug: Show the complete melody structure
                print(f"[DEBUG] Final melody has {len(synth_lead.notes)} total notes")
                if len(synth_lead.notes) > 0:
                    print(f"[DEBUG] First note: pitch={synth_lead.notes[0].pitch}, start={synth_lead.notes[0].start}, end={synth_lead.notes[0].end}")
                    print(f"[DEBUG] Last note: pitch={synth_lead.notes[-1].pitch}, start={synth_lead.notes[-1].start}, end={synth_lead.notes[-1].end}")
                    
                    # Show timing distribution
                    first_bar_notes = [n for n in synth_lead.notes if n.start < seconds_per_bar]
                    remaining_notes = [n for n in synth_lead.notes if n.start >= seconds_per_bar]
                    print(f"[DEBUG] First bar (seed): {len(first_bar_notes)} notes")
                    print(f"[DEBUG] Remaining bars (generated): {len(remaining_notes)} notes")
                
                print("[DEBUG] Lead melody generation completed successfully")
                return melody_midi
            except Exception as e:
                print(f"[DEBUG] Error creating final melody: {e}")
                import traceback
                print(f"[DEBUG] Final melody creation traceback: {traceback.format_exc()}")
                return None
        else:
            print("[DEBUG] No suitable notes found in any generated variation")
            print("[DEBUG] Attempting to create fallback melody")
            return self._create_fallback_melody(num_bars, tempo)

    def _create_fallback_melody(self, num_bars=8, tempo=120):
        """Create a simple fallback melody when AMT generation fails."""
        print("[DEBUG] Creating fallback melody")
        beats_per_bar = 4
        seconds_per_bar = 60.0 / tempo * beats_per_bar
        
        # Create a simple C major scale melody
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        synth_lead = pretty_midi.Instrument(program=81)  # Lead 2 (sawtooth)
        
        # C major scale notes (C4 to C5)
        scale_notes = [60, 62, 64, 65, 67, 69, 71, 72]
        
        # Create a simple ascending/descending pattern
        current_time = 0
        for bar in range(num_bars):
            bar_start = bar * seconds_per_bar
            
            # First half of bar: ascending
            for i in range(4):
                note_start = bar_start + (i * seconds_per_bar / 4)
                note_end = note_start + (seconds_per_bar / 8)  # Eighth notes
                pitch = scale_notes[i % len(scale_notes)]
                
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=pitch,
                    start=note_start,
                    end=note_end
                )
                synth_lead.notes.append(note)
            
            # Second half of bar: descending
            for i in range(4):
                note_start = bar_start + ((i + 4) * seconds_per_bar / 4)
                note_end = note_start + (seconds_per_bar / 8)
                pitch = scale_notes[-(i % len(scale_notes)) - 1]
                
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=pitch,
                    start=note_start,
                    end=note_end
                )
                synth_lead.notes.append(note)
        
        midi.instruments.append(synth_lead)
        print("[DEBUG] Fallback melody created successfully")
        return midi

    def preview_audio(self, midi_file, sample_rate=44100):
        """Convert MIDI to audio and return audio data for preview."""
        audio_data = self.midi_to_audio(midi_file, sample_rate)
        return audio_data

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
        print(f"[DEBUG] Starting backing track generation with chords: {chord_progression}")
        beats_per_bar = 4
        seconds_per_bar = 60.0 / tempo * beats_per_bar
        total_duration = num_bars * seconds_per_bar
        print(f"[DEBUG] Backing with chords: tempo={tempo}, key={key}, chords={chord_progression}, duration={total_duration}s")
        
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        print("[DEBUG] Created MIDI object")
        
        # Get key offset
        key_offset = self._get_key_offset(key)
        print(f"[DEBUG] Key offset for {key}: {key_offset}")
        
        # Parse chords to get note numbers
        chord_notes_list = []
        if not chord_progression:  # Handle empty chord progression
            print("Warning: Empty chord progression provided, using default C major progression")
            chord_progression = ["C", "G", "Am", "F"]  # Default progression
        
        print(f"[DEBUG] Processing {len(chord_progression)} chords")
        for chord in chord_progression:
            try:
                notes = self._parse_chord(chord, key_offset)
                chord_notes_list.append(notes)
                print(f"[DEBUG] Chord {chord} -> notes {notes}")
            except ValueError as e:
                print(f"Error parsing chord {chord}: {e}")
                # Fallback to C major
                chord_notes_list.append([60, 64, 67])
        
        print(f"[DEBUG] Parsed {len(chord_notes_list)} chords into notes")
        if not chord_notes_list:  # Double check we have at least one chord
            print("Warning: No valid chords found, using C major")
            chord_notes_list = [[60, 64, 67]]
        
        # Repeat chord progression to fill num_bars
        progression_length = len(chord_notes_list)
        print(f"[DEBUG] Chord progression length: {progression_length}")
        if progression_length == 0:  # Should never happen due to above checks, but just in case
            print("Warning: Progression length is 0, using single C major chord")
            progression_length = 1
            chord_notes_list = [[60, 64, 67]]
        
        print(f"[DEBUG] Creating full progression for {num_bars} bars")
        full_progression = (chord_notes_list * ((num_bars // progression_length) + 1))[:num_bars]
        print(f"[DEBUG] Full progression created with {len(full_progression)} chords")
        
        # Piano/Chord part
        print("[DEBUG] Creating piano part")
        piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
        for i, chord in enumerate(full_progression):
            print(f"[DEBUG] Processing piano chord {i}: {chord}")
            start_time = i * seconds_per_bar
            end_time = min(start_time + seconds_per_bar, total_duration)
            for note_pitch in chord:
                # Add root note and octave
                note = pretty_midi.Note(velocity=80, pitch=note_pitch, start=start_time, end=end_time)
                piano.notes.append(note)
                # Add octave above for fuller sound
                note = pretty_midi.Note(velocity=70, pitch=note_pitch + 12, start=start_time, end=end_time)
                piano.notes.append(note)
        midi.instruments.append(piano)
        print("[DEBUG] Piano part completed")
        
        # Bass part
        print("[DEBUG] Creating bass part")
        bass = pretty_midi.Instrument(program=32)  # Acoustic Bass
        for i, chord in enumerate(full_progression):
            print(f"[DEBUG] Processing bass chord {i}: {chord}")
            if not chord or len(chord) == 0:  # Safety check for empty chord
                print(f"[DEBUG] Skipping empty chord at position {i}")
                continue
            start_time = i * seconds_per_bar
            sixteenth_note_duration = seconds_per_bar / 16
            for j in range(16):
                note_start = start_time + (j * sixteenth_note_duration)
                note_end = min(note_start + (sixteenth_note_duration / 4), total_duration)
                beat_position = j % 4
                if beat_position == 0 and j > 0:
                    # Play root note of chord on beat
                    root_note = chord[0] - 12  # One octave below
                    note = pretty_midi.Note(velocity=90, pitch=root_note, start=note_start, end=note_end)
                    bass.notes.append(note)
                elif beat_position == 2 and len(chord) >= 3:  # Safety check for chord length
                    # Play fifth on off-beat
                    fifth_note = chord[2] - 12  # One octave below
                    note = pretty_midi.Note(velocity=70, pitch=fifth_note, start=note_start, end=note_end)
                    bass.notes.append(note)
        midi.instruments.append(bass)
        print("[DEBUG] Bass part completed")
        
        # Drums part
        print("[DEBUG] Creating drums part")
        drums = pretty_midi.Instrument(program=0, is_drum=True)
        for i in range(num_bars):
            start_time = i * seconds_per_bar
            beat_duration = seconds_per_bar / 4
            
            # Kick drum on beats 1 and 3
            kick_note = pretty_midi.Note(velocity=100, pitch=36, start=start_time, end=min(start_time + 0.1, total_duration))
            drums.notes.append(kick_note)
            kick_note = pretty_midi.Note(velocity=100, pitch=36, start=start_time + 2*beat_duration, end=min(start_time + 2*beat_duration + 0.1, total_duration))
            drums.notes.append(kick_note)
            
            # Snare on beats 2 and 4
            snare_note = pretty_midi.Note(velocity=90, pitch=38, start=start_time + beat_duration, end=min(start_time + beat_duration + 0.1, total_duration))
            drums.notes.append(snare_note)
            snare_note = pretty_midi.Note(velocity=90, pitch=38, start=start_time + 3*beat_duration, end=min(start_time + 3*beat_duration + 0.1, total_duration))
            drums.notes.append(snare_note)
            
            # Hi-hat on every eighth note
            eighth_duration = seconds_per_bar / 8
            for j in range(8):
                hat_start = start_time + j * eighth_duration
                hat_note = pretty_midi.Note(velocity=60, pitch=42, start=hat_start, end=min(hat_start + 0.05, total_duration))
                drums.notes.append(hat_note)
        
        midi.instruments.append(drums)
        print("[DEBUG] Drums part completed")
        
        # Ensure the MIDI file has the correct end time
        print("[DEBUG] Adjusting MIDI times")
        try:
            # Instead of using adjust_times, we'll set the end time directly
            midi.tick_to_time = lambda tick: tick * 60.0 / (tempo * midi.resolution)
            midi.time_to_tick = lambda time: int(time * midi.resolution * tempo / 60.0)
            # Set the end time by adjusting the last note's end time
            for instrument in midi.instruments:
                if instrument.notes:
                    last_note = max(instrument.notes, key=lambda x: x.end)
                    if last_note.end < total_duration:
                        last_note.end = total_duration
            print("[DEBUG] MIDI times adjusted successfully")
        except Exception as e:
            print(f"[DEBUG] Error adjusting MIDI times: {e}")
            # If time adjustment fails, we'll still return the MIDI file
            # The notes will end at their natural end times
            pass
        
        print("[DEBUG] Backing track generation completed")
        return midi

    def create_seed_melody_from_notes(self, notes: list[dict], key: str = "C", num_bars: int = 8, tempo: int = 120):
        """Create a seed melody from user-provided notes.
        
        Args:
            notes: List of note dictionaries with 'pitch', 'start', 'duration' keys
                  pitch: MIDI note number (0-127)
                  start: Start time in beats (0.0, 1.0, 2.0, etc.)
                  duration: Duration in beats (0.25, 0.5, 1.0, etc.)
            key: Key signature for diatonic validation
            num_bars: Total number of bars
            tempo: Tempo in BPM
        """
        beats_per_bar = 4
        seconds_per_bar = 60.0 / tempo * beats_per_bar
        total_duration = num_bars * seconds_per_bar
        print(f"[DEBUG] Seed from notes: tempo={tempo}, key={key}, notes={len(notes)}")
        
        seed_midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        synth_lead = pretty_midi.Instrument(program=81)  # Lead 2 (sawtooth)
        
        # Get key offset for diatonic validation
        key_offset = self._get_key_offset(key)
        
        # Define diatonic scale for the key
        diatonic_intervals = [0, 2, 4, 5, 7, 9, 11]  # Major scale intervals
        base_note = 60 + key_offset  # C4 + key offset
        diatonic_scale = [base_note + interval for interval in diatonic_intervals]
        
        # Convert user notes to MIDI notes
        for note_data in notes:
            pitch = note_data.get('pitch', 60)
            start_beats = note_data.get('start', 0.0)
            duration_beats = note_data.get('duration', 1.0)
            
            # Convert beats to seconds
            start_time = start_beats * seconds_per_bar / 4  # 4 beats per bar
            duration = duration_beats * seconds_per_bar / 4
            end_time = start_time + duration
            
            # Ensure timing is within bounds
            if start_time >= total_duration:
                continue
            end_time = min(end_time, total_duration)
            
            # Validate pitch is in diatonic scale (with some tolerance)
            if pitch not in diatonic_scale:
                # Find closest diatonic note
                closest_pitch = min(diatonic_scale, key=lambda x: abs(x - pitch))
                pitch = closest_pitch
                print(f"[DEBUG] Adjusted pitch to diatonic: {pitch}")
            
            # Create the note
            note = pretty_midi.Note(
                velocity=90,
                pitch=pitch,
                start=start_time,
                end=end_time
            )
            synth_lead.notes.append(note)
            print(f"[DEBUG] Seed note: pitch={pitch}, start={start_time}s, end={end_time}s")
        
        seed_midi.instruments.append(synth_lead)
        
        # Ensure the MIDI file has the correct end time
        seed_midi.adjust_times(0, total_duration)
        return seed_midi

    def generate_lead_melody_with_seed(self, backing_midi, seed_melody=None, num_bars=8, tempo=120):
        """Generate a lead melody using AMT over the backing track with an optional custom seed melody."""
        print("Converting backing track to events...")
        beats_per_bar = 4
        seconds_per_bar = 60.0 / tempo * beats_per_bar
        
        # Use custom seed melody if provided, otherwise create default
        if seed_melody is None:
            seed_melody = self.create_seed_melody(key=60, num_bars=num_bars, tempo=tempo)
            print("Using default seed melody")
        else:
            print(f"Using custom seed melody with {len(seed_melody.instruments[0].notes)} notes")
        
        # Save the seed melody for reference
        seed_path = self.output_dir / 'seed_melody.mid'
        seed_melody.write(str(seed_path))
        print(f"Saved seed melody to {seed_path}")

        temp_seed_file = self.output_dir / 'temp_seed.mid'
        seed_melody.write(str(temp_seed_file))
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
                try:
                    print(f"[DEBUG] Converting variation {i+1} to MIDI with {len(generated_events)} events")
                    generated_midi = events_to_midi(generated_events)
                    print(f"[DEBUG] Successfully converted variation {i+1} to MIDI")
                    
                    # Debug: Let's see what we got (skip the instruments check since it's a MidiFile, not PrettyMIDI)
                    print(f"[DEBUG] Generated MIDI file created successfully")
                    
                except Exception as e:
                    print(f"[DEBUG] Error converting variation {i+1} to MIDI: {e}")
                    import traceback
                    print(f"[DEBUG] MIDI conversion traceback: {traceback.format_exc()}")
                    continue

                # Convert to pretty_midi for easier manipulation
                temp_file = self.output_dir / f'temp_generated_{i}.mid'
                try:
                    print(f"[DEBUG] Saving variation {i+1} to temporary file")
                    generated_midi.save(str(temp_file))
                    print(f"[DEBUG] Loading variation {i+1} into pretty_midi")
                    pretty_generated = pretty_midi.PrettyMIDI(str(temp_file))
                    print(f"[DEBUG] Successfully loaded variation {i+1} into pretty_midi")
                except Exception as e:
                    print(f"[DEBUG] Error processing variation {i+1} MIDI: {e}")
                    import traceback
                    print(f"[DEBUG] MIDI processing traceback: {traceback.format_exc()}")
                    continue
                finally:
                    try:
                        os.remove(temp_file)
                        print(f"[DEBUG] Cleaned up temporary file for variation {i+1}")
                    except Exception as e:
                        print(f"[DEBUG] Warning: Could not remove temporary file for variation {i+1}: {e}")

                # Process the generated melody
                try:
                    print(f"[DEBUG] Processing notes for variation {i+1}")
                    valid_notes = []
                    current_time = 0

                    # Sort all notes by start time
                    all_notes = []
                    for instrument in pretty_generated.instruments:
                        if not instrument.is_drum:
                            all_notes.extend(instrument.notes)
                    all_notes.sort(key=lambda x: x.start)
                    print(f"[DEBUG] Found {len(all_notes)} notes in variation {i+1}")

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

                    print(f"[DEBUG] Filtered to {len(valid_notes)} valid notes in variation {i+1}")
                    if len(valid_notes) > 0:
                        print(f"[DEBUG] First valid note: pitch={valid_notes[0].pitch}, start={valid_notes[0].start}, end={valid_notes[0].end}")
                        print(f"[DEBUG] Last valid note: pitch={valid_notes[-1].pitch}, start={valid_notes[-1].start}, end={valid_notes[-1].end}")

                    # Select the variation with the most notes (most interesting)
                    if len(valid_notes) > best_note_count:
                        best_note_count = len(valid_notes)
                        best_melody = valid_notes
                        print(f"[DEBUG] New best variation found with {best_note_count} notes")
                except Exception as e:
                    print(f"[DEBUG] Error processing notes for variation {i+1}: {e}")
                    import traceback
                    print(f"[DEBUG] Note processing traceback: {traceback.format_exc()}")
                    continue

            if best_melody:
                # Create a new MIDI file for the melody
                melody_midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
                synth_lead = pretty_midi.Instrument(program=81)  # Lead 2 (sawtooth)
                synth_lead.notes = best_melody
                melody_midi.instruments.append(synth_lead)
                return melody_midi
            else:
                print("No suitable notes found in any generated variation")
                print("[DEBUG] Attempting to create fallback melody")
                return self._create_fallback_melody(num_bars, tempo)

        except Exception as e:
            print(f"Generation failed: {e}")
            return None 