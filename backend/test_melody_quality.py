#!/usr/bin/env python3
"""
Test script to verify melody generation quality and analyze generated melodies.
"""

import pretty_midi
import numpy as np
from app.core.generator import AMTBackingGenerator

def analyze_melody(midi_file, name="Melody"):
    """Analyze a melody for musical quality."""
    print(f"\n=== {name} Analysis ===")
    
    if len(midi_file.instruments) == 0:
        print("No instruments found")
        return
    
    instrument = midi_file.instruments[0]
    notes = instrument.notes
    
    if len(notes) == 0:
        print("No notes found")
        return
    
    print(f"Number of notes: {len(notes)}")
    
    # Analyze pitch range
    pitches = [note.pitch for note in notes]
    print(f"Pitch range: {min(pitches)} - {max(pitches)} ({min(pitches)}-{max(pitches)})")
    
    # Analyze timing
    start_times = [note.start for note in notes]
    end_times = [note.end for note in notes]
    durations = [note.end - note.start for note in notes]
    
    print(f"Duration: {min(start_times):.2f}s - {max(end_times):.2f}s")
    print(f"Average note duration: {np.mean(durations):.3f}s")
    print(f"Note duration range: {min(durations):.3f}s - {max(durations):.3f}s")
    
    # Analyze melodic intervals
    intervals = []
    for i in range(1, len(pitches)):
        interval = abs(pitches[i] - pitches[i-1])
        intervals.append(interval)
    
    if intervals:
        print(f"Average melodic interval: {np.mean(intervals):.1f} semitones")
        print(f"Interval range: {min(intervals)} - {max(intervals)} semitones")
        
        # Check for stepwise motion (intervals of 1-2 semitones)
        stepwise = sum(1 for interval in intervals if interval <= 2)
        print(f"Stepwise motion: {stepwise}/{len(intervals)} ({stepwise/len(intervals)*100:.1f}%)")
    
    # Analyze velocity (dynamics)
    velocities = [note.velocity for note in notes]
    print(f"Average velocity: {np.mean(velocities):.1f}")
    print(f"Velocity range: {min(velocities)} - {max(velocities)}")
    
    # Check for monophonic melody
    overlapping_notes = 0
    for i in range(len(notes)):
        for j in range(i+1, len(notes)):
            if (notes[i].start < notes[j].end and notes[j].start < notes[i].end):
                overlapping_notes += 1
    
    if overlapping_notes == 0:
        print("✓ Monophonic melody (no overlapping notes)")
    else:
        print(f"⚠ Polyphonic melody ({overlapping_notes} overlapping note pairs)")
    
    # Check for reasonable note density
    total_duration = max(end_times) - min(start_times)
    notes_per_second = len(notes) / total_duration if total_duration > 0 else 0
    print(f"Note density: {notes_per_second:.2f} notes/second")
    
    if 0.5 <= notes_per_second <= 4.0:
        print("✓ Reasonable note density")
    else:
        print(f"⚠ Unusual note density: {notes_per_second:.2f} notes/second")

def test_melody_generation():
    """Test melody generation with different parameters."""
    print("Testing Melody Generation Quality")
    print("=" * 50)
    
    # Initialize generator
    generator = AMTBackingGenerator()
    
    # Test 1: Basic backing track and melody
    print("\nTest 1: Basic C major backing track")
    backing = generator.create_backing_track(num_bars=4, tempo=120)
    analyze_melody(backing, "Backing Track")
    
    lead = generator.generate_lead_melody(backing, num_bars=4, tempo=120)
    if lead:
        analyze_melody(lead, "Generated Lead Melody")
    else:
        print("Failed to generate lead melody")
    
    # Test 2: Custom chord progression
    print("\nTest 2: Custom chord progression (C - Am - F - G)")
    backing_chords = generator.create_backing_track_with_chords(
        chord_progression=["C", "Am", "F", "G"],
        key="C",
        num_bars=4,
        tempo=120
    )
    analyze_melody(backing_chords, "Chord Backing Track")
    
    lead_chords = generator.generate_lead_melody(backing_chords, num_bars=4, tempo=120)
    if lead_chords:
        analyze_melody(lead_chords, "Generated Lead Melody (Chords)")
    else:
        print("Failed to generate lead melody for chord progression")
    
    # Test 3: Different key
    print("\nTest 3: G major backing track")
    backing_g = generator.create_backing_track_with_chords(
        chord_progression=["G", "Em", "C", "D"],
        key="G",
        num_bars=4,
        tempo=120
    )
    analyze_melody(backing_g, "G Major Backing Track")
    
    lead_g = generator.generate_lead_melody(backing_g, num_bars=4, tempo=120)
    if lead_g:
        analyze_melody(lead_g, "Generated Lead Melody (G Major)")
    else:
        print("Failed to generate lead melody for G major")
    
    print("\n" + "=" * 50)
    print("Melody Generation Test Complete")

if __name__ == "__main__":
    test_melody_generation() 