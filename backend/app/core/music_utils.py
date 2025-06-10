"""
Music Theory Utilities Module
============================

This module provides music theory utilities for chord parsing and key handling.
"""

def parse_chord(chord_name: str, key_offset: int = 0) -> list[int]:
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

def get_key_offset(key: str) -> int:
    """Get the semitone offset for a given key signature."""
    key_offsets = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
        'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
    }
    
    if key not in key_offsets:
        raise ValueError(f"Unknown key: {key}")
    
    return key_offsets[key]

def get_diatonic_scale(key: str) -> list[int]:
    """Get the diatonic major scale for a given key."""
    key_offset = get_key_offset(key)
    diatonic_intervals = [0, 2, 4, 5, 7, 9, 11]  # Major scale intervals
    base_note = 60 + key_offset  # C4 + key offset
    diatonic_scale = [base_note + interval for interval in diatonic_intervals]
    return diatonic_scale

def get_num_bars_for_duration(duration_seconds: float, tempo: int, beats_per_bar: int = 4) -> tuple[int, float]:
    """Helper to calculate the number of bars needed for a given duration and tempo."""
    import numpy as np
    seconds_per_bar = 60.0 / tempo * beats_per_bar
    return int(np.ceil(duration_seconds / seconds_per_bar)), seconds_per_bar 