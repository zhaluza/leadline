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

    def generate_lead_melody(self, backing_midi, duration_seconds=16):
        """Generate a lead melody using AMT over the backing track."""
        print("Converting backing track to events...")
        
        # Save temporary MIDI file
        temp_file = self.output_dir / 'temp_backing.mid'
        backing_midi.write(str(temp_file))
        
        # Convert to events
        seed_events = midi_to_events(str(temp_file))
        os.remove(temp_file)
        
        print(f"Generating lead melody with {len(seed_events)} seed events...")
        
        try:
            # Generate with seed conditioning - removed temperature parameter
            generated_events = generate(
                self.model,
                start_time=0,
                end_time=duration_seconds,
                controls=seed_events,
                top_p=0.8
            )
            
            # Convert back to MIDI
            generated_midi = events_to_midi(generated_events)
            
            # Convert to pretty_midi for easier manipulation
            temp_file = self.output_dir / 'temp_generated.mid'
            generated_midi.save(str(temp_file))
            pretty_generated = pretty_midi.PrettyMIDI(str(temp_file))
            os.remove(temp_file)
            
            # Filter for melody notes (higher pitch range)
            melody_midi = pretty_midi.PrettyMIDI()
            
            for instrument in pretty_generated.instruments:
                if not instrument.is_drum:
                    # Only keep notes in melody range
                    melody_notes = [note for note in instrument.notes if 60 <= note.pitch <= 90]
                    if melody_notes:
                        instrument.notes = melody_notes
                        melody_midi.instruments.append(instrument)
            
            return melody_midi
            
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