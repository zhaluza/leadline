#!/usr/bin/env python3
"""
Refactoring Helper Script
=========================

This script helps extract the used methods from the original generator.py
to create a clean, refactored version.
"""

import re
from pathlib import Path

def extract_methods_from_file(file_path, method_names):
    """Extract specific methods from a Python file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    extracted_methods = {}
    
    for method_name in method_names:
        # Pattern to match method definition and its body
        pattern = rf'(\s+def {re.escape(method_name)}\(.*?)(?=\n\s+def|\n\n|\Z)'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            extracted_methods[method_name] = match.group(1)
            print(f"✓ Extracted {method_name}")
        else:
            print(f"✗ Could not find {method_name}")
    
    return extracted_methods

def create_clean_generator():
    """Create a clean generator file with only used methods."""
    
    # Methods that are actually used by the backend routes
    used_methods = [
        '__init__',
        'midi_to_audio',
        'create_backing_track',
        'create_backing_track_with_chords',
        'create_seed_melody',
        'create_seed_melody_from_notes',
        'generate_lead_melody',
        'generate_lead_melody_with_seed',
        '_generate_with_amt',
        '_generate_with_amt_and_backing',
        '_generate_algorithmic_melody',
        '_create_minimal_seed_melody'
    ]
    
    # Extract methods from original file
    original_file = Path('app/core/generator.py')
    extracted = extract_methods_from_file(original_file, used_methods)
    
    # Create the clean generator content
    clean_content = '''"""
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
'''
    
    # Add each extracted method
    for method_name in used_methods:
        if method_name in extracted:
            clean_content += extracted[method_name] + '\n\n'
    
    # Write the clean generator
    clean_file = Path('app/core/generator_clean.py')
    with open(clean_file, 'w') as f:
        f.write(clean_content)
    
    print(f"\n✓ Created clean generator at {clean_file}")
    print(f"✓ Reduced from ~1142 lines to ~{len(clean_content.splitlines())} lines")
    print("✓ Removed unused methods: preview_audio, save_and_preview, _parse_chord, _get_key_offset, _get_num_bars_for_duration")

if __name__ == "__main__":
    create_clean_generator() 