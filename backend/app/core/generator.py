"""
AMT Backing Generator Core Module (Clean)
========================================

This module provides the core AMTBackingGenerator class for generating backing tracks
and lead melodies. This is a clean version that removes unused methods and
separates concerns into utility modules.
"""

import torch
from transformers import AutoModelForCausalLM
from anticipation.sample import generate
from anticipation.convert import events_to_midi, midi_to_events
import pretty_midi
import numpy as np
import os
from pathlib import Path

from .audio_utils import AudioConverter
from .music_utils import parse_chord, get_key_offset, get_diatonic_scale, get_num_bars_for_duration

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
        
        # Initialize audio converter
        self.audio_converter = AudioConverter(soundfont_path)
        
        # Create output directory if it doesn't exist
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

    def midi_to_audio(self, midi_file, sample_rate=44100):
        """Convert MIDI to audio using the audio converter."""
        return self.audio_converter.midi_to_audio(midi_file, sample_rate)

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

    def create_backing_track_with_chords(self, chord_progression: list[str], key: str = "C", num_bars: int = 8, tempo: int = 120):
        """Create a backing track with custom chord progression including drums, bass, and piano."""
        print(f"[DEBUG] Starting backing track generation with chords: {chord_progression}")
        beats_per_bar = 4
        seconds_per_bar = 60.0 / tempo * beats_per_bar
        total_duration = num_bars * seconds_per_bar
        print(f"[DEBUG] Backing with chords: tempo={tempo}, key={key}, chords={chord_progression}, duration={total_duration}s")
        
        # Create MIDI object
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        print("[DEBUG] Created MIDI object")
        
        # Get key offset
        key_offset = get_key_offset(key)
        print(f"[DEBUG] Key offset for {key}: {key_offset}")
        
        # Parse chords into notes
        chord_notes = []
        print(f"[DEBUG] Processing {len(chord_progression)} chords")
        for chord_name in chord_progression:
            notes = parse_chord(chord_name, key_offset)
            chord_notes.append(notes)
            print(f"[DEBUG] Chord {chord_name} -> notes {notes}")
        
        print(f"[DEBUG] Parsed {len(chord_notes)} chords into notes")
        
        # Create full progression for all bars
        chord_progression_length = len(chord_notes)
        print(f"[DEBUG] Chord progression length: {chord_progression_length}")
        
        full_progression = []
        print("[DEBUG] Creating full progression for {num_bars} bars")
        for bar in range(num_bars):
            chord_index = bar % chord_progression_length
            full_progression.append(chord_notes[chord_index])
        print(f"[DEBUG] Full progression created with {len(full_progression)} chords")
        
        # Create piano part
        piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
        print("[DEBUG] Creating piano part")
        for i, chord in enumerate(full_progression):
            bar_start = i * seconds_per_bar
            print(f"[DEBUG] Processing piano chord {i}: {chord}")
            
            for note_pitch in chord:
                note = pretty_midi.Note(
                    velocity=75,
                    pitch=note_pitch,
                    start=bar_start,
                    end=bar_start + seconds_per_bar
                )
                piano.notes.append(note)
        print("[DEBUG] Piano part completed")
        
        # Create bass part
        bass = pretty_midi.Instrument(program=32)  # Acoustic Bass
        print("[DEBUG] Creating bass part")
        for i, chord in enumerate(full_progression):
            bar_start = i * seconds_per_bar
            root_note = chord[0] - 12  # One octave lower
            print(f"[DEBUG] Processing bass chord {i}: {chord}")
            
            note = pretty_midi.Note(
                velocity=80,
                pitch=root_note,
                start=bar_start,
                end=bar_start + seconds_per_bar
            )
            bass.notes.append(note)
        print("[DEBUG] Bass part completed")
        
        # Create drums part
        drums = pretty_midi.Instrument(program=0, is_drum=True)
        print("[DEBUG] Creating drums part")
        for bar in range(num_bars):
            bar_start = bar * seconds_per_bar
            
            # Kick drum on beats 1 and 3
            for beat in [0, 2]:
                note = pretty_midi.Note(
                    velocity=80,
                    pitch=36,  # Bass Drum 1
                    start=bar_start + beat * seconds_per_bar / 4,
                    end=bar_start + (beat + 0.5) * seconds_per_bar / 4
                )
                drums.notes.append(note)
            
            # Snare on beats 2 and 4
            for beat in [1, 3]:
                note = pretty_midi.Note(
                    velocity=70,
                    pitch=38,  # Acoustic Snare
                    start=bar_start + beat * seconds_per_bar / 4,
                    end=bar_start + (beat + 0.5) * seconds_per_bar / 4
                )
                drums.notes.append(note)
            
            # Hi-hat on all beats
            for beat in range(4):
                note = pretty_midi.Note(
                    velocity=60,
                    pitch=42,  # Closed Hi-Hat
                    start=bar_start + beat * seconds_per_bar / 4,
                    end=bar_start + (beat + 0.25) * seconds_per_bar / 4
                )
                drums.notes.append(note)
        print("[DEBUG] Drums part completed")
        
        midi.instruments.extend([piano, bass, drums])
        
        # Adjust MIDI times
        print("[DEBUG] Adjusting MIDI times")
        # Ensure the MIDI file is at least total_duration long
        for instrument in midi.instruments:
            if instrument.notes:
                instrument.notes[-1].end = max(instrument.notes[-1].end, total_duration)
        print("[DEBUG] MIDI times adjusted successfully")
        print("[DEBUG] Backing track generation completed")
        
        return midi

    def create_seed_melody(self, key=60, num_bars=8, tempo=120):
        """Create a seed melody for just the first measure using random eighth notes."""
        beats_per_bar = 4
        seconds_per_bar = 60.0 / tempo * beats_per_bar
        print(f"[DEBUG] Seed: tempo={tempo}, beats_per_bar={beats_per_bar}, seconds_per_bar={seconds_per_bar}")

        # Create a MIDI file
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        synth_lead = pretty_midi.Instrument(program=81)  # Lead 2 (sawtooth)

        # Get diatonic scale for the key
        if isinstance(key, int):
            # If key is a MIDI note number, convert to note name
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key_name = note_names[key % 12]
        else:
            key_name = key
        
        scale = get_diatonic_scale(key_name)
        print(f"[DEBUG] Generated scale with {len(scale)} notes: {scale}")
        
        # Calculate eighth note duration
        eighth_note_duration = seconds_per_bar / 8  # 8 eighth notes per bar
        
        # Generate eighth notes for the first measure only
        current_time = 0
        for _ in range(8):  # 8 eighth notes in a measure
            # Randomly select a note from the scale
            scale_index = np.random.randint(0, len(scale))
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

    def create_seed_melody_from_notes(self, notes: list[dict], key: str = "C", num_bars: int = 8, tempo: int = 120):
        """Create a seed melody from user-provided notes.
        
        Args:
            notes: List of note dictionaries with 'pitch', 'start', and 'duration' keys
            key: Key signature (e.g., "C", "G", "F#")
            num_bars: Number of bars for the seed melody
            tempo: Tempo in BPM
        """
        beats_per_bar = 4
        seconds_per_bar = 60.0 / tempo * beats_per_bar
        total_duration = num_bars * seconds_per_bar
        print(f"[DEBUG] Seed from notes: tempo={tempo}, key={key}, notes={len(notes)}")
        
        seed_midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        synth_lead = pretty_midi.Instrument(program=81)  # Lead 2 (sawtooth)
        
        # Get key offset for diatonic validation
        key_offset = get_key_offset(key)
        
        # Define diatonic scale for the key
        diatonic_scale = get_diatonic_scale(key)
        
        # Process each note
        for note_data in notes:
            pitch = note_data['pitch']
            start_beats = note_data['start']
            duration_beats = note_data['duration']
            
            # Convert beats to seconds
            start_time = start_beats * seconds_per_bar / 4
            end_time = start_time + (duration_beats * seconds_per_bar / 4)
            
            # Validate pitch is in diatonic scale (with some tolerance)
            if pitch not in diatonic_scale:
                # Find closest diatonic note
                closest_pitch = min(diatonic_scale, key=lambda x: abs(x - pitch))
                if abs(closest_pitch - pitch) <= 2:  # Within 2 semitones
                    pitch = closest_pitch
                    print(f"[DEBUG] Adjusted pitch to diatonic scale: {pitch}")
            
            # Ensure pitch is in reasonable range
            if pitch < 60:
                pitch += 12
            elif pitch > 84:
                pitch -= 12
            
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

    def generate_lead_melody(self, backing_midi, num_bars=8, tempo=120):
        """Generate a lead melody using AMT over the backing track and a seed melody."""
        print("[DEBUG] Starting lead melody generation")
        
        # Extract tempo from backing track if available
        backing_tempo = tempo  # Default to passed tempo
        try:
            tempo_times, tempo_values = backing_midi.get_tempo_changes()
            if len(tempo_values) > 0:
                backing_tempo = int(tempo_values[0])
                print(f"[DEBUG] Extracted tempo from backing track: {backing_tempo} BPM")
            else:
                print(f"[DEBUG] Using passed tempo: {tempo} BPM")
        except Exception as e:
            print(f"[DEBUG] Could not extract tempo from backing track: {e}")
            print(f"[DEBUG] Using passed tempo: {tempo} BPM")
        
        beats_per_bar = 4
        seconds_per_bar = 60.0 / backing_tempo * beats_per_bar
        print(f"[DEBUG] Lead melody params: num_bars={num_bars}, tempo={backing_tempo}, seconds_per_bar={seconds_per_bar}")
        
        # Try AMT generation first, but fall back to algorithmic generation if it fails
        try:
            print("[DEBUG] Attempting AMT-based generation...")
            amt_melody = self._generate_with_amt(backing_midi, num_bars, backing_tempo)
            if amt_melody and len(amt_melody.instruments[0].notes) > 0:
                print("[DEBUG] AMT generation successful")
                return amt_melody
            else:
                print("[DEBUG] AMT generation produced no notes, using algorithmic fallback")
        except Exception as e:
            print(f"[DEBUG] AMT generation failed: {e}")
            print("[DEBUG] Using algorithmic fallback")
        
        # Fall back to algorithmic melody generation
        return self._generate_algorithmic_melody(backing_midi, num_bars, backing_tempo)

    def generate_lead_melody_with_seed(self, backing_midi, seed_melody=None, num_bars=8, tempo=120):
        """Generate a lead melody using AMT over the backing track with an optional custom seed melody."""
        print("Converting backing track to events...")
        
        # Extract tempo from backing track if available
        backing_tempo = tempo  # Default to passed tempo
        try:
            tempo_times, tempo_values = backing_midi.get_tempo_changes()
            if len(tempo_values) > 0:
                backing_tempo = int(tempo_values[0])
                print(f"[DEBUG] Extracted tempo from backing track: {backing_tempo} BPM")
            else:
                print(f"[DEBUG] Using passed tempo: {tempo} BPM")
        except Exception as e:
            print(f"[DEBUG] Could not extract tempo from backing track: {e}")
            print(f"[DEBUG] Using passed tempo: {tempo} BPM")
        
        beats_per_bar = 4
        seconds_per_bar = 60.0 / backing_tempo * beats_per_bar
        
        # Try AMT generation first, but fall back to algorithmic generation if it fails
        try:
            print("[DEBUG] Attempting AMT-based generation with custom seed...")
            amt_melody = self._generate_with_amt_and_backing(backing_midi, seed_melody, num_bars, backing_tempo)
            if amt_melody and len(amt_melody.instruments[0].notes) > 0:
                print("[DEBUG] AMT generation successful")
                return amt_melody
            else:
                print("[DEBUG] AMT generation produced no notes, using algorithmic fallback")
        except Exception as e:
            print(f"[DEBUG] AMT generation failed: {e}")
            print("[DEBUG] Using algorithmic fallback")
        
        # Fall back to algorithmic melody generation
        return self._generate_algorithmic_melody(backing_midi, num_bars, backing_tempo)

    def _generate_with_amt(self, backing_midi, num_bars=8, tempo=120):
        """Attempt AMT-based generation with robust error handling."""
        beats_per_bar = 4
        seconds_per_bar = 60.0 / tempo * beats_per_bar
        
        # Create a minimal seed melody
        seed_midi = self._create_minimal_seed_melody(key=60, tempo=tempo)
        
        # Save the seed melody for reference
        seed_path = self.output_dir / 'seed_melody.mid'
        try:
            seed_midi.write(str(seed_path))
            print(f"[DEBUG] Saved seed melody to {seed_path}")
        except Exception as e:
            print(f"[DEBUG] Error saving seed melody: {e}")
            return None

        # Convert seed melody to events with aggressive filtering
        print("[DEBUG] Converting seed melody to events")
        temp_seed_file = self.output_dir / 'temp_seed.mid'
        try:
            seed_midi.write(str(temp_seed_file))
            raw_seed_events = midi_to_events(str(temp_seed_file))
            
            # Very aggressive filtering - only keep very small values
            seed_events = []
            for event in raw_seed_events:
                if isinstance(event, int) and 0 <= event < 100:  # Very conservative limit
                    seed_events.append(event)
            
            print(f"[DEBUG] Filtered to {len(seed_events)} valid seed events")
            
        except Exception as e:
            print(f"[DEBUG] Error converting seed melody to events: {e}")
            seed_events = []
        finally:
            try:
                os.remove(temp_seed_file)
            except:
                pass

        if not seed_events:
            print("[DEBUG] No valid seed events, trying without controls")
            seed_events = []

        try:
            # Use the actual backing track duration for generation
            generation_duration = num_bars * seconds_per_bar
            
            print(f"[DEBUG] Attempting AMT generation for {generation_duration:.2f} seconds at tempo {tempo}")
            generated_events = generate(
                self.model,
                start_time=0,
                end_time=generation_duration,
                controls=seed_events if seed_events else None,
                top_p=0.7,  # Very conservative
                delta=50    # Very small delta
            )
            
            print(f"[DEBUG] Generated {len(generated_events)} events")
            
            # Very aggressive filtering
            filtered_events = []
            for event in generated_events:
                if isinstance(event, int) and 0 <= event < 100:  # Very conservative limit
                    filtered_events.append(event)
            
            print(f"[DEBUG] Filtered from {len(generated_events)} to {len(filtered_events)} valid events")
            
            if len(filtered_events) < 3:  # Need at least 3 events for a meaningful melody
                print("[DEBUG] Too few valid events, using algorithmic fallback")
                return None
            
            # Try to convert to MIDI
            try:
                generated_midi = events_to_midi(filtered_events)
                print("[DEBUG] Successfully converted to MIDI")
            except Exception as e:
                print(f"[DEBUG] MIDI conversion failed: {e}")
                return None
            
            # Convert to pretty_midi and process
            temp_file = self.output_dir / 'temp_generated.mid'
            try:
                generated_midi.save(str(temp_file))
                pretty_generated = pretty_midi.PrettyMIDI(str(temp_file))
                
                # Process the generated melody
                valid_notes = []
                current_time = 0
                
                all_notes = []
                for instrument in pretty_generated.instruments:
                    if not instrument.is_drum:
                        all_notes.extend(instrument.notes)
                all_notes.sort(key=lambda x: x.start)
                
                # Filter for monophonic melody in guitar range
                for note in all_notes:
                    if 60 <= note.pitch <= 84:  # Guitar range
                        start_time = max(0, note.start)
                        end_time = min(num_bars * seconds_per_bar, note.end)
                        
                        if end_time <= start_time or start_time < current_time:
                            continue
                        
                        duration = min(0.5, max(0.125, end_time - start_time))
                        velocity = min(120, max(80, note.velocity))
                        
                        new_note = pretty_midi.Note(
                            velocity=velocity,
                            pitch=note.pitch,
                            start=start_time,
                            end=start_time + duration
                        )
                        valid_notes.append(new_note)
                        current_time = new_note.end
                
                print(f"[DEBUG] Processed {len(valid_notes)} valid notes from AMT")
                
                if len(valid_notes) == 0:
                    print("[DEBUG] No valid notes processed, using algorithmic fallback")
                    return None
                
                # Create final MIDI with proper tempo
                final_midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
                synth_lead = pretty_midi.Instrument(program=81)  # Lead 2 (sawtooth)
                
                for note in valid_notes:
                    synth_lead.notes.append(note)
                
                final_midi.instruments.append(synth_lead)
                
                # Clean up temp file
                try:
                    os.remove(temp_file)
                except:
                    pass
                
                print(f"[DEBUG] AMT generation successful with {len(valid_notes)} notes")
                return final_midi
                
            except Exception as e:
                print(f"[DEBUG] Error processing generated MIDI: {e}")
                return None
                
        except Exception as e:
            print(f"[DEBUG] AMT generation failed: {e}")
            return None

    def _generate_with_amt_and_backing(self, backing_midi, seed_melody=None, num_bars=8, tempo=120):
        """Attempt AMT-based generation with backing track and custom seed."""
        beats_per_bar = 4
        seconds_per_bar = 60.0 / tempo * beats_per_bar
        
        # Use custom seed melody if provided, otherwise create minimal seed
        if seed_melody is None:
            seed_melody = self._create_minimal_seed_melody(key=60, tempo=tempo)
            print("Using minimal seed melody")
        else:
            print(f"Using custom seed melody with {len(seed_melody.instruments[0].notes)} notes")
        
        # Save the seed melody for reference
        seed_path = self.output_dir / 'seed_melody.mid'
        seed_melody.write(str(seed_path))
        print(f"Saved seed melody to {seed_path}")

        # Convert seed melody to events with aggressive filtering
        temp_seed_file = self.output_dir / 'temp_seed.mid'
        seed_melody.write(str(temp_seed_file))
        raw_seed_events = midi_to_events(str(temp_seed_file))
        os.remove(temp_seed_file)
        
        # Filter seed events to stay within anticipation library limits
        seed_events = []
        for event in raw_seed_events:
            if isinstance(event, int) and 0 <= event < 100:  # Very conservative limit
                seed_events.append(event)
        
        print(f"[DEBUG] Filtered to {len(seed_events)} valid seed events")

        # Convert backing track to events with filtering
        temp_backing_file = self.output_dir / 'temp_backing.mid'
        backing_midi.write(str(temp_backing_file))
        raw_backing_events = midi_to_events(str(temp_backing_file))
        os.remove(temp_backing_file)
        
        # Filter backing events
        backing_events = []
        for event in raw_backing_events:
            if isinstance(event, int) and 0 <= event < 100:  # Very conservative limit
                backing_events.append(event)
        
        print(f"[DEBUG] Filtered to {len(backing_events)} valid backing events")

        # Use a subset of seed events for conditioning (every other event)
        seed_subset = []
        for i, event in enumerate(seed_events):
            if i % 2 == 0:  # Every other event
                seed_subset.append(event)

        # Combine subset of seed events with backing events
        combined_events = backing_events + seed_subset

        print(f"Generating lead melody with {len(combined_events)} conditioning events...")

        try:
            # Use the actual backing track duration for generation
            generation_duration = num_bars * seconds_per_bar
            
            # Generate with combined seed conditioning
            generated_events = generate(
                self.model,
                start_time=0,
                end_time=generation_duration,
                controls=combined_events,
                top_p=0.7,  # Very conservative
                delta=50    # Very small delta
            )

            print(f"[DEBUG] Generated {len(generated_events)} events")
            
            # Very aggressive filtering
            filtered_events = []
            for event in generated_events:
                if isinstance(event, int) and 0 <= event < 100:  # Very conservative limit
                    filtered_events.append(event)
            
            print(f"[DEBUG] Filtered from {len(generated_events)} to {len(filtered_events)} valid events")
            
            if len(filtered_events) < 3:  # Need at least 3 events for a meaningful melody
                print("[DEBUG] Too few valid events, using algorithmic fallback")
                return None
            
            # Try to convert to MIDI
            try:
                generated_midi = events_to_midi(filtered_events)
                print("[DEBUG] Successfully converted to MIDI")
            except Exception as e:
                print(f"[DEBUG] MIDI conversion failed: {e}")
                return None

            # Convert to pretty_midi and process
            temp_file = self.output_dir / 'temp_generated.mid'
            try:
                generated_midi.save(str(temp_file))
                pretty_generated = pretty_midi.PrettyMIDI(str(temp_file))
                
                # Process the generated melody
                valid_notes = []
                current_time = 0
                
                all_notes = []
                for instrument in pretty_generated.instruments:
                    if not instrument.is_drum:
                        all_notes.extend(instrument.notes)
                all_notes.sort(key=lambda x: x.start)
                
                # Filter for monophonic melody in guitar range
                for note in all_notes:
                    if 60 <= note.pitch <= 84:  # Guitar range
                        start_time = max(0, note.start)
                        end_time = min(num_bars * seconds_per_bar, note.end)
                        
                        if end_time <= start_time or start_time < current_time:
                            continue
                        
                        duration = min(0.5, max(0.125, end_time - start_time))
                        velocity = min(120, max(80, note.velocity))
                        
                        new_note = pretty_midi.Note(
                            velocity=velocity,
                            pitch=note.pitch,
                            start=start_time,
                            end=start_time + duration
                        )
                        valid_notes.append(new_note)
                        current_time = new_note.end
                
                print(f"[DEBUG] Processed {len(valid_notes)} valid notes from AMT")
                
                if len(valid_notes) == 0:
                    print("[DEBUG] No valid notes processed, using algorithmic fallback")
                    return None
                
                # Create final MIDI with proper tempo
                final_midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
                synth_lead = pretty_midi.Instrument(program=81)  # Lead 2 (sawtooth)
                
                for note in valid_notes:
                    synth_lead.notes.append(note)
                
                final_midi.instruments.append(synth_lead)
                
                # Clean up temp file
                try:
                    os.remove(temp_file)
                except:
                    pass
                
                print(f"[DEBUG] AMT generation successful with {len(valid_notes)} notes")
                return final_midi
                
            except Exception as e:
                print(f"[DEBUG] Error processing generated MIDI: {e}")
                return None
                
        except Exception as e:
            print(f"[DEBUG] AMT generation failed: {e}")
            return None

    def _generate_algorithmic_melody(self, backing_midi, num_bars=8, tempo=120):
        """Generate a melody using algorithmic approach when AMT fails."""
        print("[DEBUG] Generating algorithmic melody")
        beats_per_bar = 4
        seconds_per_bar = 60.0 / tempo * beats_per_bar
        
        # Create a MIDI file
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        synth_lead = pretty_midi.Instrument(program=81)  # Lead 2 (sawtooth)
        
        # Define the major scale (C major)
        scale = [60, 62, 64, 65, 67, 69, 71, 72]  # C4 to C5
        
        # Create a melodic pattern that follows common melodic rules
        current_time = 0
        current_pitch = 60  # Start on C4
        
        for bar in range(num_bars):
            bar_start = bar * seconds_per_bar
            
            # Generate 4 notes per bar (quarter notes)
            for beat in range(4):
                note_start = bar_start + (beat * seconds_per_bar / 4)
                note_end = note_start + (seconds_per_bar / 8)  # Eighth notes for more interest
                
                # Choose next note based on melodic rules
                if beat == 0:  # First beat of bar - often root, third, or fifth
                    possible_pitches = [scale[0], scale[2], scale[4]]  # C, E, G
                elif beat == 2:  # Third beat - often a different chord tone
                    possible_pitches = [scale[1], scale[3], scale[5]]  # D, F, A
                else:  # Other beats - more freedom
                    possible_pitches = scale
                
                # Add some variation and avoid too much repetition
                if np.random.random() < 0.3:  # 30% chance to step up or down
                    if current_pitch in scale:
                        current_idx = scale.index(current_pitch)
                        if np.random.random() < 0.5 and current_idx > 0:
                            current_pitch = scale[current_idx - 1]  # Step down
                        elif current_idx < len(scale) - 1:
                            current_pitch = scale[current_idx + 1]  # Step up
                    else:
                        current_pitch = np.random.choice(possible_pitches)
                else:
                    current_pitch = np.random.choice(possible_pitches)
                
                # Ensure we stay in a reasonable range
                if current_pitch < 60:
                    current_pitch += 12
                elif current_pitch > 84:
                    current_pitch -= 12
                
                # Add some rhythmic variation
                if np.random.random() < 0.2:  # 20% chance for sixteenth note
                    note_end = note_start + (seconds_per_bar / 16)
                elif np.random.random() < 0.1:  # 10% chance for dotted eighth
                    note_end = note_start + (seconds_per_bar / 8 * 1.5)
                
                # Create the note
                note = pretty_midi.Note(
                    velocity=np.random.randint(85, 115),  # Varied velocity
                    pitch=current_pitch,
                    start=note_start,
                    end=note_end
                )
                synth_lead.notes.append(note)
                
                current_time = note_end
        
        midi.instruments.append(synth_lead)
        print(f"[DEBUG] Algorithmic melody created with {len(synth_lead.notes)} notes")
        return midi

    def _create_minimal_seed_melody(self, key=60, tempo=120):
        """Create a minimal seed melody (just 2 beats) for better AMT conditioning."""
        print("[DEBUG] Creating minimal seed melody")
        beats_per_bar = 4
        seconds_per_bar = 60.0 / tempo * beats_per_bar
        beat_duration = seconds_per_bar / 4
        
        # Create a MIDI file
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        synth_lead = pretty_midi.Instrument(program=81)  # Lead 2 (sawtooth)

        # Define a simple 2-beat pattern using the major scale
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
        
        # Create just 2 quarter notes for the seed
        current_time = 0
        
        # First note: root
        note1 = pretty_midi.Note(
            velocity=90,
            pitch=scale[0],  # Root note
            start=current_time,
            end=current_time + beat_duration
        )
        synth_lead.notes.append(note1)
        print(f"[DEBUG] Seed note 1: pitch={scale[0]}, start={current_time}, end={current_time + beat_duration}")
        
        current_time += beat_duration
        
        # Second note: third
        note2 = pretty_midi.Note(
            velocity=90,
            pitch=scale[2],  # Third note
            start=current_time,
            end=current_time + beat_duration
        )
        synth_lead.notes.append(note2)
        print(f"[DEBUG] Seed note 2: pitch={scale[2]}, start={current_time}, end={current_time + beat_duration}")

        midi.instruments.append(synth_lead)
        return midi

