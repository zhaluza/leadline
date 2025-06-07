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
        """Create a simple backing track with drums, bass, and piano chords."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        
        # Add piano track for chords (using a lighter voicing)
        piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
        
        # Simple chord progression: C - Am - F - G (in key of C major)
        # Using a lighter voicing with fewer notes
        chord_progression = [
            [60, 64, 67],  # C major (C4, E4, G4)
            [57, 60, 64],  # A minor (A3, C4, E4)
            [53, 57, 60],  # F major (F3, A3, C4)
            [55, 59, 62],  # G major (G3, B3, D4)
        ]
        
        # Add chord notes with slightly lower velocity for a softer sound
        for i, chord in enumerate(chord_progression):
            start_time = i * (duration_seconds / 4)
            end_time = start_time + (duration_seconds / 4) - 0.1
            
            for note_pitch in chord:
                note = pretty_midi.Note(
                    velocity=60,  # Reduced velocity for softer piano
                    pitch=note_pitch,
                    start=start_time,
                    end=end_time
                )
                piano.notes.append(note)
        
        midi.instruments.append(piano)
        
        # Add bass track with more interesting pattern
        bass = pretty_midi.Instrument(program=33)  # Electric Bass
        bass_notes = [48, 45, 41, 43]  # Root notes of chord progression
        
        for i, note_pitch in enumerate(bass_notes):
            start_time = i * (duration_seconds / 4)
            
            # Play root note on beat 1
            note = pretty_midi.Note(
                velocity=100,  # Increased velocity for stronger bass
                pitch=note_pitch,
                start=start_time,
                end=start_time + 1.0
            )
            bass.notes.append(note)
            
            # Play fifth on beat 2
            fifth_pitch = note_pitch + 7
            fifth_note = pretty_midi.Note(
                velocity=90,
                pitch=fifth_pitch,
                start=start_time + 1.0,
                end=start_time + 2.0
            )
            bass.notes.append(fifth_note)
            
            # Play root note again on beat 3
            note = pretty_midi.Note(
                velocity=95,
                pitch=note_pitch,
                start=start_time + 2.0,
                end=start_time + 3.0
            )
            bass.notes.append(note)
            
            # Play octave on beat 4
            octave_pitch = note_pitch + 12
            octave_note = pretty_midi.Note(
                velocity=85,
                pitch=octave_pitch,
                start=start_time + 3.0,
                end=start_time + 4.0
            )
            bass.notes.append(octave_note)
        
        midi.instruments.append(bass)
        
        # Add drum track with more pronounced pattern
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
                
                # Hi-hat on all beats (slightly softer)
                hihat = pretty_midi.Note(velocity=70, pitch=42, start=beat_time, end=beat_time + 0.05)
                drums.notes.append(hihat)
                
                # Add crash cymbal on beat 1 of each measure
                if beat == 0:
                    crash = pretty_midi.Note(velocity=100, pitch=49, start=beat_time, end=beat_time + 0.1)
                    drums.notes.append(crash)
        
        midi.instruments.append(drums)
        return midi

    def create_seed_melody(self, key=60, duration_seconds=16):
        """Create a seed melody using common guitar patterns in the given key."""
        # Create a MIDI file for the seed melody
        seed_midi = pretty_midi.PrettyMIDI()
        
        # Create a guitar instrument (using a clean electric guitar sound)
        guitar = pretty_midi.Instrument(program=27)  # Clean Electric Guitar
        
        # Define a pentatonic scale in the given key
        # Using A minor pentatonic as an example (A, C, D, E, G)
        scale = [key, key + 3, key + 5, key + 7, key + 10]
        
        # Create a simple seed melody with common guitar patterns
        # Pattern 1: Ascending scale run
        for i, pitch in enumerate(scale):
            note = pretty_midi.Note(
                velocity=80,
                pitch=pitch,
                start=i * 0.25,  # Quarter notes
                end=(i + 1) * 0.25
            )
            guitar.notes.append(note)
        
        # Pattern 2: Call and response (higher register)
        # Call
        call_notes = [scale[0] + 12, scale[2] + 12, scale[4] + 12, scale[2] + 12]
        for i, pitch in enumerate(call_notes):
            note = pretty_midi.Note(
                velocity=85,
                pitch=pitch,
                start=2 + i * 0.5,  # Half notes
                end=2 + (i + 1) * 0.5
            )
            guitar.notes.append(note)
        
        # Response (lower register)
        response_notes = [scale[4], scale[2], scale[0], scale[2]]
        for i, pitch in enumerate(response_notes):
            note = pretty_midi.Note(
                velocity=80,
                pitch=pitch,
                start=4 + i * 0.5,
                end=4 + (i + 1) * 0.5
            )
            guitar.notes.append(note)
        
        # Pattern 3: Blues-style phrase
        blues_notes = [scale[0], scale[0] + 1, scale[2], scale[4], scale[2], scale[0]]
        for i, pitch in enumerate(blues_notes):
            duration = 0.25 if i in [1, 3] else 0.5  # Shorter notes for passing tones
            start_time = 6 + sum(0.25 if j in [1, 3] else 0.5 for j in range(i))
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
        temp_seed_file = self.output_dir / 'temp_seed.mid'
        seed_midi.write(str(temp_seed_file))
        seed_events = midi_to_events(str(temp_seed_file))
        os.remove(temp_seed_file)
        
        # Convert backing track to events
        temp_backing_file = self.output_dir / 'temp_backing.mid'
        backing_midi.write(str(temp_backing_file))
        backing_events = midi_to_events(str(temp_backing_file))
        os.remove(temp_backing_file)
        
        # Combine seed and backing events
        combined_events = seed_events + backing_events
        
        print(f"Generating lead melody with {len(combined_events)} seed events...")
        
        try:
            # Generate with combined seed conditioning
            generated_events = generate(
                self.model,
                start_time=0,
                end_time=duration_seconds,
                controls=combined_events,
                top_p=0.7  # Slightly higher for more variation while maintaining structure
            )
            
            # Convert back to MIDI
            generated_midi = events_to_midi(generated_events)
            
            # Convert to pretty_midi for easier manipulation
            temp_file = self.output_dir / 'temp_generated.mid'
            generated_midi.save(str(temp_file))
            pretty_generated = pretty_midi.PrettyMIDI(str(temp_file))
            os.remove(temp_file)
            
            # Create a new MIDI file for the melody
            melody_midi = pretty_midi.PrettyMIDI()
            
            # Create a guitar lead instrument
            guitar_lead = pretty_midi.Instrument(program=27)  # Clean Electric Guitar
            
            # Process and filter the generated notes
            valid_notes = []
            for instrument in pretty_generated.instruments:
                if not instrument.is_drum:
                    # Filter notes to be in guitar range (E2-E6)
                    for note in instrument.notes:
                        # Keep notes in guitar range (40-88)
                        if 40 <= note.pitch <= 88:
                            # Ensure valid timing
                            start_time = max(0, note.start)
                            end_time = min(duration_seconds, note.end)
                            
                            # Skip notes with invalid timing
                            if end_time <= start_time:
                                continue
                                
                            # Adjust velocity for better dynamics
                            velocity = min(100, max(60, note.velocity))
                            
                            # Create a new note with adjusted parameters
                            new_note = pretty_midi.Note(
                                velocity=velocity,
                                pitch=note.pitch,
                                start=start_time,
                                end=min(end_time, start_time + 0.75)  # Slightly longer notes for guitar
                            )
                            valid_notes.append(new_note)
            
            # Sort notes by start time
            valid_notes.sort(key=lambda x: x.start)
            
            # Add note smoothing with timing validation
            if valid_notes:
                smoothed_notes = []
                for i in range(len(valid_notes)):
                    current_note = valid_notes[i]
                    
                    # Skip if note duration is too short
                    if current_note.end - current_note.start < 0.1:  # Longer minimum duration for guitar
                        continue
                    
                    # Adjust end time if next note is too close
                    if i < len(valid_notes) - 1:
                        next_note = valid_notes[i + 1]
                        if next_note.start - current_note.end < 0.1:
                            # Ensure we don't create negative duration
                            new_end = max(current_note.start + 0.1, next_note.start - 0.1)
                            current_note.end = new_end
                    
                    smoothed_notes.append(current_note)
                
                # Add all valid, smoothed notes to the guitar lead
                guitar_lead.notes = smoothed_notes
                melody_midi.instruments.append(guitar_lead)
                return melody_midi
            else:
                print("No suitable notes found in the generated melody")
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