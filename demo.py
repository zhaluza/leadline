#!/usr/bin/env python3
"""
AMT Backing Track Generator Demo

This script demonstrates how to use the AMT Backing Track Generator to create
backing tracks and generate lead melodies.

First, make sure you have installed all dependencies from requirements.txt:
    pip install -r requirements.txt
"""

from amt_backing_generator import AMTBackingGenerator
import pretty_midi

def main():
    # Initialize the generator
    print("Initializing AMT Backing Generator...")
    generator = AMTBackingGenerator()
    
    # Generate backing track
    print("\nGenerating backing track...")
    backing_track = generator.create_backing_track()
    print("Backing track generated. Saving and creating preview...")
    backing_audio = generator.save_and_preview(backing_track, "backing_track.mid")
    print("Backing track saved to output/backing_track.mid")
    
    # Generate lead melody
    print("\nGenerating lead melody...")
    lead_melody = generator.generate_lead_melody(backing_track)
    
    if lead_melody:
        print("Lead melody generated. Saving and creating preview...")
        lead_audio = generator.save_and_preview(lead_melody, "lead_melody.mid")
        print("Lead melody saved to output/lead_melody.mid")
        
        # Create and save combined track
        print("\nCreating combined track...")
        combined = pretty_midi.PrettyMIDI()
        for instrument in backing_track.instruments:
            combined.instruments.append(instrument)
        for instrument in lead_melody.instruments:
            combined.instruments.append(instrument)
        
        print("Saving combined track...")
        combined_audio = generator.save_and_preview(combined, "combined_track.mid")
        print("Combined track saved to output/combined_track.mid")
        
        print("\nDemo complete! All files have been saved to the output directory.")
        print("You can find:")
        print("- output/backing_track.mid (backing track with drums, bass, and chords)")
        print("- output/lead_melody.mid (AI-generated lead melody)")
        print("- output/combined_track.mid (full track with everything combined)")
    else:
        print("Failed to generate lead melody")

if __name__ == "__main__":
    main() 