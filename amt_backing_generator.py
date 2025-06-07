import torch
from transformers import AutoModelForCausalLM
from anticipation.sample import generate
from anticipation.convert import events_to_midi, midi_to_events
import pretty_midi
import numpy as np
from IPython.display import Audio
import os
from pathlib import Path

class AMTBackingGenerator:
    def __init__(self, model_name='stanford-crfm/music-medium-800k'):
        """Initialize the AMT backing generator with the specified model."""
        print("Loading AMT model...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        print(f"Model loaded on {self.device}")
        
        # Create output directory if it doesn't exist
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

    def create_backing_track(self, duration_seconds=16, tempo=120):
        """Create a synthwave-style backing track with drums, bass, and synth chords."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        
        # Add synth pad for chords (using a warm pad sound)
        synth_pad = pretty_midi.Instrument(program=89)  # Warm Pad
        
        # Synthwave chord progression: Am - F - C - G (in key of A minor)
        chord_progression = [
            [57, 60, 64],  # A minor (A3, C4, E4)
            [53, 57, 60],  # F major (F3, A3, C4)
            [48, 52, 55],  # C major (C3, E3, G3)
            [55, 59, 62],  # G major (G3, B3, D4)
        ]
        
        # Add arpeggiated chords with synthwave style
        for i, chord in enumerate(chord_progression):
            start_time = i * (duration_seconds / 4)
            # Arpeggiate each chord
            for j, note_pitch in enumerate(chord):
                # Add some variation to the arpeggiation timing
                note_start = start_time + (j * 0.125)  # 16th notes
                note_end = note_start + 0.25  # 8th notes
                
                # Add the main chord note
                note = pretty_midi.Note(
                    velocity=70,
                    pitch=note_pitch,
                    start=note_start,
                    end=note_end
                )
                synth_pad.notes.append(note)
                
                # Add an octave up for more synthwave feel
                note = pretty_midi.Note(
                    velocity=65,
                    pitch=note_pitch + 12,
                    start=note_start,
                    end=note_end
                )
                synth_pad.notes.append(note)
        
        midi.instruments.append(synth_pad)
        
        # Add synth bass with a more interesting pattern
        synth_bass = pretty_midi.Instrument(program=87)  # Lead 8 (bass + lead)
        bass_notes = [57, 53, 48, 55]  # Root notes of chord progression
        
        for i, note_pitch in enumerate(bass_notes):
            start_time = i * (duration_seconds / 4)
            
            # Main bass note on beat 1
            note = pretty_midi.Note(
                velocity=100,
                pitch=note_pitch,
                start=start_time,
                end=start_time + 0.5
            )
            synth_bass.notes.append(note)
            
            # Add a higher note on beat 2
            higher_note = pretty_midi.Note(
                velocity=90,
                pitch=note_pitch + 7,  # Perfect fifth
                start=start_time + 0.5,
                end=start_time + 1.0
            )
            synth_bass.notes.append(higher_note)
            
            # Add a lower note on beat 3
            lower_note = pretty_midi.Note(
                velocity=95,
                pitch=note_pitch - 12,  # Octave down
                start=start_time + 1.0,
                end=start_time + 1.5
            )
            synth_bass.notes.append(lower_note)
            
            # Add a slide up to the root on beat 4
            slide_note = pretty_midi.Note(
                velocity=85,
                pitch=note_pitch - 5,  # Perfect fourth
                start=start_time + 1.5,
                end=start_time + 2.0
            )
            synth_bass.notes.append(slide_note)
        
        midi.instruments.append(synth_bass)
        
        # Add drum track with synthwave style
        drums = pretty_midi.Instrument(program=0, is_drum=True)
        beats_per_measure = 4
        measures = 4
        beat_duration = duration_seconds / (beats_per_measure * measures)
        
        for measure in range(measures):
            for beat in range(beats_per_measure):
                beat_time = measure * beats_per_measure * beat_duration + beat * beat_duration
                
                # Kick drum on beats 1 and 3 (stronger)
                if beat in [0, 2]:
                    kick = pretty_midi.Note(velocity=120, pitch=36, start=beat_time, end=beat_time + 0.1)
                    drums.notes.append(kick)
                
                # Snare on beats 2 and 4 (stronger)
                if beat in [1, 3]:
                    snare = pretty_midi.Note(velocity=110, pitch=38, start=beat_time, end=beat_time + 0.1)
                    drums.notes.append(snare)
                
                # Hi-hat on all beats with variation
                if beat in [0, 2]:  # Stronger on beats 1 and 3
                    hihat = pretty_midi.Note(velocity=80, pitch=42, start=beat_time, end=beat_time + 0.05)
                    drums.notes.append(hihat)
                else:  # Softer on beats 2 and 4
                    hihat = pretty_midi.Note(velocity=60, pitch=42, start=beat_time, end=beat_time + 0.05)
                    drums.notes.append(hihat)
                
                # Add closed hi-hat on off-beats for more groove
                if beat not in [0, 2]:
                    off_beat_time = beat_time + beat_duration/2
                    hihat = pretty_midi.Note(velocity=50, pitch=42, start=off_beat_time, end=off_beat_time + 0.05)
                    drums.notes.append(hihat)
                
                # Add crash cymbal on beat 1 of each measure
                if beat == 0:
                    crash = pretty_midi.Note(velocity=100, pitch=49, start=beat_time, end=beat_time + 0.1)
                    drums.notes.append(crash)
        
        midi.instruments.append(drums)
        return midi

    def create_seed_melody(self, key=57, duration_seconds=16):
        """Create a seed melody using common guitar patterns in the given key."""
        # Create a MIDI file for the seed melody
        seed_midi = pretty_midi.PrettyMIDI()
        
        # Create a guitar instrument (using a clean electric guitar sound)
        guitar = pretty_midi.Instrument(program=27)  # Clean Electric Guitar
        
        # Define a minor pentatonic scale with some additional notes for more interest
        # Using A minor pentatonic with added notes (A, C, D, E, G, B, F#)
        scale = [key, key + 3, key + 5, key + 7, key + 10, key + 12, key + 14]
        
        # Create a more interesting seed melody with common guitar patterns
        # Pattern 1: Ascending run with added notes
        for i, pitch in enumerate(scale[:5]):
            note = pretty_midi.Note(
                velocity=80,
                pitch=pitch,
                start=i * 0.125,  # 16th notes
                end=(i + 1) * 0.125
            )
            guitar.notes.append(note)
        
        # Pattern 2: Call and response with more movement
        # Call (higher register with more notes)
        call_notes = [scale[0] + 12, scale[2] + 12, scale[4] + 12, scale[6] + 12, scale[4] + 12, scale[2] + 12]
        for i, pitch in enumerate(call_notes):
            duration = 0.25 if i in [2, 4] else 0.125  # Mix of 8th and 16th notes
            start_time = 1 + sum(0.25 if j in [2, 4] else 0.125 for j in range(i))
            note = pretty_midi.Note(
                velocity=85,
                pitch=pitch,
                start=start_time,
                end=start_time + duration
            )
            guitar.notes.append(note)
        
        # Response (lower register with more movement)
        response_notes = [scale[4], scale[2], scale[0], scale[1], scale[2], scale[4]]
        for i, pitch in enumerate(response_notes):
            duration = 0.25 if i in [0, 3] else 0.125
            start_time = 3 + sum(0.25 if j in [0, 3] else 0.125 for j in range(i))
            note = pretty_midi.Note(
                velocity=80,
                pitch=pitch,
                start=start_time,
                end=start_time + duration
            )
            guitar.notes.append(note)
        
        # Pattern 3: Blues-style phrase with more notes
        blues_notes = [scale[0], scale[0] + 1, scale[2], scale[4], scale[6], scale[4], scale[2], scale[0]]
        for i, pitch in enumerate(blues_notes):
            duration = 0.125 if i in [1, 3, 5] else 0.25  # Mix of 8th and 16th notes
            start_time = 5 + sum(0.125 if j in [1, 3, 5] else 0.25 for j in range(i))
            note = pretty_midi.Note(
                velocity=90,
                pitch=pitch,
                start=start_time,
                end=start_time + duration
            )
            guitar.notes.append(note)
        
        seed_midi.instruments.append(guitar)
        return seed_midi

    def generate_lead_melody(self, backing_midi, duration_seconds=16):
        """Generate a lead melody using AMT over the backing track."""
        print("Converting backing track to events...")
        
        # Create and convert seed melody to events
        print("Creating seed melody for better generation...")
        seed_midi = self.create_seed_melody(key=57)  # A minor
        
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
        
        # Instead of combining all events, use a subset of seed events
        # This reduces the direct influence of the seed melody
        seed_subset = []
        for event in seed_events:
            # Only keep every third event from the seed melody
            # This provides some guidance without being too prescriptive
            if len(seed_subset) % 3 == 0:
                seed_subset.append(event)
        
        # Combine subset of seed events with backing events
        # This gives more weight to the backing track
        combined_events = backing_events + seed_subset
        
        print(f"Generating lead melody with {len(combined_events)} conditioning events...")
        
        try:
            # Generate multiple variations and select the most interesting one
            num_variations = 3
            best_melody = None
            best_note_count = 0
            
            for i in range(num_variations):
                # Vary the top_p parameter for more creative generation
                current_top_p = 0.8 + (i * 0.1)  # 0.8, 0.9, 1.0
                
                # Generate with combined seed conditioning
                generated_events = generate(
                    self.model,
                    start_time=0,
                    end_time=duration_seconds,
                    controls=combined_events,
                    top_p=current_top_p
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
                
                # Sort all notes by start time first
                all_notes = []
                for instrument in pretty_generated.instruments:
                    if not instrument.is_drum:
                        all_notes.extend(instrument.notes)
                all_notes.sort(key=lambda x: x.start)
                
                # Filter for monophonic melody with more variety
                for note in all_notes:
                    # Keep notes in guitar range (40-88)
                    if 40 <= note.pitch <= 88:
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
                        duration = min(0.5, max(0.0625, end_time - start_time))  # 16th to 8th notes
                        if i == 0:  # First variation: more staccato
                            duration *= 0.75
                        elif i == 1:  # Second variation: more legato
                            duration *= 1.25
                        
                        # Adjust velocity for better dynamics with more variation
                        base_velocity = min(100, max(60, note.velocity))
                        if i == 0:  # First variation: more dynamic
                            velocity = int(base_velocity * (0.8 + (0.4 * (note.pitch % 12) / 12)))
                        elif i == 1:  # Second variation: more consistent
                            velocity = int(base_velocity * 1.1)
                        else:
                            velocity = base_velocity
                        
                        # Create a new note with adjusted parameters
                        new_note = pretty_midi.Note(
                            velocity=velocity,
                            pitch=note.pitch,
                            start=start_time,
                            end=start_time + duration
                        )
                        valid_notes.append(new_note)
                        current_time = new_note.end
                
                # Add note smoothing with timing validation
                if valid_notes:
                    smoothed_notes = []
                    for j in range(len(valid_notes)):
                        current_note = valid_notes[j]
                        
                        # Skip if note duration is too short
                        if current_note.end - current_note.start < 0.1:
                            continue
                        
                        # Adjust end time if next note is too close
                        if j < len(valid_notes) - 1:
                            next_note = valid_notes[j + 1]
                            if next_note.start - current_note.end < 0.1:
                                # Ensure we don't create negative duration
                                new_end = max(current_note.start + 0.1, next_note.start - 0.1)
                                current_note.end = new_end
                        
                        smoothed_notes.append(current_note)
                    
                    # Select the variation with the most notes (most interesting)
                    if len(smoothed_notes) > best_note_count:
                        best_note_count = len(smoothed_notes)
                        best_melody = smoothed_notes
            
            if best_melody:
                # Create a new MIDI file for the melody
                melody_midi = pretty_midi.PrettyMIDI()
                guitar_lead = pretty_midi.Instrument(program=27)  # Clean Electric Guitar
                guitar_lead.notes = best_melody
                melody_midi.instruments.append(guitar_lead)
                return melody_midi
            else:
                print("No suitable notes found in any generated variation")
                return None
            
        except Exception as e:
            print(f"Generation failed: {e}")
            return None

    def preview_audio(self, midi_file, sample_rate=44100):
        """Convert MIDI to audio and return an IPython Audio object for preview."""
        audio_data = midi_file.synthesize(fs=sample_rate)
        return Audio(audio_data, rate=sample_rate)

    def save_and_preview(self, midi_file, filename):
        """Save MIDI file and return audio preview."""
        output_path = self.output_dir / filename
        midi_file.write(str(output_path))
        print(f"Saved to {output_path}")
        return self.preview_audio(midi_file)

def main():
    # Initialize generator
    generator = AMTBackingGenerator()
    
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