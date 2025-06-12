"""
Audio Utilities Module
=====================

This module provides audio conversion utilities for the AMT application.
"""

import numpy as np
import os
import subprocess
import tempfile
from pathlib import Path
import pretty_midi

class AudioConverter:
    """Handles MIDI to audio conversion using fluidsynth or fallback methods."""
    
    def __init__(self, soundfont_path=None):
        """Initialize the audio converter with optional soundfont."""
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
        
        # Check if fluidsynth is installed
        self._check_fluidsynth()
    
    def _check_fluidsynth(self):
        """Check if fluidsynth is available on the system."""
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
    
    def midi_to_audio(self, midi_file, sample_rate=44100):
        """Convert MIDI to audio using fluidsynth and the specified soundfont."""
        # Extract tempo from MIDI file
        target_tempo = 120  # Default tempo
        try:
            tempo_times, tempo_values = midi_file.get_tempo_changes()
            if len(tempo_values) > 0:
                target_tempo = float(tempo_values[0])
                print(f"[DEBUG] Extracted tempo from MIDI: {target_tempo} BPM")
        except Exception as e:
            print(f"[DEBUG] Could not extract tempo from MIDI: {e}")
        
        if self.soundfont_path is None:
            print("No soundfont specified, falling back to pretty_midi synthesizer...")
            audio_data = midi_file.synthesize(fs=sample_rate)
            return self._adjust_audio_tempo(audio_data, target_tempo, sample_rate, midi_file)
        
        if not os.path.exists(self.soundfont_path):
            print(f"Soundfont not found at {self.soundfont_path}, falling back to pretty_midi synthesizer...")
            audio_data = midi_file.synthesize(fs=sample_rate)
            return self._adjust_audio_tempo(audio_data, target_tempo, sample_rate, midi_file)
        
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
            
            # Adjust tempo by resampling
            return self._adjust_audio_tempo(audio_data, target_tempo, sample_rate, midi_file)
            
        except subprocess.CalledProcessError as e:
            print(f"Error running fluidsynth: {e}")
            print(f"Command output: {e.stdout}")
            print(f"Error output: {e.stderr}")
            print("Falling back to pretty_midi synthesizer...")
            audio_data = midi_file.synthesize(fs=sample_rate)
            return self._adjust_audio_tempo(audio_data, target_tempo, sample_rate, midi_file)
        except Exception as e:
            print(f"Unexpected error: {e}")
            print("Falling back to pretty_midi synthesizer...")
            audio_data = midi_file.synthesize(fs=sample_rate)
            return self._adjust_audio_tempo(audio_data, target_tempo, sample_rate, midi_file)
        finally:
            # Clean up temporary files
            try:
                os.unlink(temp_audio_path)
                os.unlink(temp_midi_path)
                print("Cleaned up temporary files")
            except Exception as e:
                print(f"Error cleaning up temporary files: {e}")
    
    def _adjust_audio_tempo(self, audio_data, target_tempo, sample_rate, midi_file=None):
        """Adjust audio tempo by resampling to match expected duration from MIDI tempo and bars."""
        # Calculate expected duration from MIDI file if available
        beats_per_bar = 4
        num_bars = 1
        if midi_file is not None:
            # Try to infer number of bars from the MIDI notes
            notes = []
            for inst in midi_file.instruments:
                notes.extend(inst.notes)
            if notes:
                last_note_end = max(note.end for note in notes)
                # Calculate number of bars from last note end and tempo
                seconds_per_bar = 60.0 / target_tempo * beats_per_bar
                num_bars = max(1, int(np.ceil(last_note_end / seconds_per_bar)))
        
        expected_duration = 60.0 / target_tempo * beats_per_bar * num_bars
        actual_duration = len(audio_data) / sample_rate
        duration_ratio = actual_duration / expected_duration
        print(f"[DEBUG] Audio duration check: expected={expected_duration:.2f}s, actual={actual_duration:.2f}s, ratio={duration_ratio:.3f}")
        
        # If the duration ratio is close to 1.0, the synthesizer is already playing at the correct tempo
        if abs(duration_ratio - 1.0) < 0.05:
            print(f"[DEBUG] Synthesizer already playing at correct tempo, no adjustment needed")
            return audio_data
        
        # Resample the audio to match the expected duration
        from scipy import signal
        new_length = int(len(audio_data) / duration_ratio)
        print(f"[DEBUG] Resampling audio: {len(audio_data)} -> {new_length} samples to match expected duration")
        resampled_audio = signal.resample(audio_data, new_length)
        return resampled_audio

    def midi_to_audio_no_adjustment(self, midi_file, sample_rate=44100):
        """Convert MIDI to audio without tempo adjustment (for seed melodies)."""
        # Extract tempo from MIDI file (for logging only)
        target_tempo = 120  # Default tempo
        try:
            tempo_times, tempo_values = midi_file.get_tempo_changes()
            if len(tempo_values) > 0:
                target_tempo = float(tempo_values[0])
                print(f"[DEBUG] Extracted tempo from MIDI: {target_tempo} BPM (no adjustment)")
        except Exception as e:
            print(f"[DEBUG] Could not extract tempo from MIDI: {e}")
        
        if self.soundfont_path is None:
            print("No soundfont specified, falling back to pretty_midi synthesizer...")
            audio_data = midi_file.synthesize(fs=sample_rate)
            return audio_data  # No adjustment
        
        if not os.path.exists(self.soundfont_path):
            print(f"Soundfont not found at {self.soundfont_path}, falling back to pretty_midi synthesizer...")
            audio_data = midi_file.synthesize(fs=sample_rate)
            return audio_data  # No adjustment
        
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
            
            # No tempo adjustment for seed melodies
            return audio_data
            
        except subprocess.CalledProcessError as e:
            print(f"Error running fluidsynth: {e}")
            print(f"Command output: {e.stdout}")
            print(f"Error output: {e.stderr}")
            print("Falling back to pretty_midi synthesizer...")
            audio_data = midi_file.synthesize(fs=sample_rate)
            return audio_data  # No adjustment
        except Exception as e:
            print(f"Unexpected error: {e}")
            print("Falling back to pretty_midi synthesizer...")
            audio_data = midi_file.synthesize(fs=sample_rate)
            return audio_data  # No adjustment
        finally:
            # Clean up temporary files
            try:
                os.unlink(temp_audio_path)
                os.unlink(temp_midi_path)
                print("Cleaned up temporary files")
            except Exception as e:
                print(f"Error cleaning up temporary files: {e}") 