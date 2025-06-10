# AMT Generator Refactoring Summary

## Analysis Results

Based on my analysis of the codebase, here's what I found regarding method usage and recommendations for refactoring:

## Methods Currently Used by Backend Routes

The following methods are actively used by the backend API routes:

### Core Generation Methods (KEEP)

- `create_backing_track()` - Used by `/api/backing`
- `create_backing_track_with_chords()` - Used by `/api/backing/chords` (frontend uses this)
- `create_seed_melody()` - Used by `/api/seed` and `/api/lead/{id}/preview-seed`
- `create_seed_melody_from_notes()` - Used by `/api/lead/{id}/custom`
- `generate_lead_melody()` - Used by `/api/lead/{id}`
- `generate_lead_melody_with_seed()` - Used by `/api/lead/{id}/custom`
- `midi_to_audio()` - Used by all routes for audio generation

### Private Helper Methods (KEEP)

- `_generate_with_amt()` - Called by `generate_lead_melody()`
- `_generate_with_amt_and_backing()` - Called by `generate_lead_melody_with_seed()`
- `_generate_algorithmic_melody()` - Fallback method for AMT failures
- `_create_minimal_seed_melody()` - Used by AMT generation methods

## Methods NOT Used (CAN BE REMOVED)

The following methods are not used by any backend routes or frontend:

### Unused Public Methods (REMOVE)

- `preview_audio()` - Not called anywhere
- `save_and_preview()` - Not called anywhere
- `_get_num_bars_for_duration()` - Moved to music_utils.py

### Unused Private Methods (REMOVE)

- `_parse_chord()` - Moved to music_utils.py
- `_get_key_offset()` - Moved to music_utils.py

## Frontend API Usage

The frontend only uses these endpoints:

- `/api/backing/chords` - For backing track generation
- `/api/lead/{id}/preview-seed` - For seed melody preview
- `/api/lead/{id}` - For lead melody generation (default)
- `/api/lead/{id}/custom` - For lead melody with custom seed

## Recommended Refactoring

### 1. Create Utility Modules (COMPLETED)

I've already created two utility modules:

#### `audio_utils.py`

- Contains `AudioConverter` class
- Handles all MIDI to audio conversion logic
- Includes fluidsynth setup and fallback handling

#### `music_utils.py`

- Contains music theory utilities:
  - `parse_chord()` - Chord name parsing
  - `get_key_offset()` - Key signature handling
  - `get_diatonic_scale()` - Scale generation
  - `get_num_bars_for_duration()` - Duration calculations

### 2. Create Clean Generator (RECOMMENDED)

Create a new `generator_clean.py` that:

#### Keeps Only Used Methods:

- `__init__()` - Model initialization
- `midi_to_audio()` - Audio conversion (delegates to AudioConverter)
- `create_backing_track()` - Basic backing track
- `create_backing_track_with_chords()` - Chord-based backing track
- `create_seed_melody()` - Seed melody generation
- `create_seed_melody_from_notes()` - Custom seed melody
- `generate_lead_melody()` - Lead melody generation
- `generate_lead_melody_with_seed()` - Lead melody with custom seed
- `_generate_with_amt()` - AMT generation attempt
- `_generate_with_amt_and_backing()` - AMT with backing track
- `_generate_algorithmic_melody()` - Algorithmic fallback
- `_create_minimal_seed_melody()` - Minimal seed for AMT

#### Removes Unused Methods:

- `preview_audio()` - Not used
- `save_and_preview()` - Not used
- `_parse_chord()` - Moved to music_utils
- `_get_key_offset()` - Moved to music_utils
- `_get_num_bars_for_duration()` - Moved to music_utils

### 3. Benefits of Refactoring

#### Reduced File Size

- Original: ~1142 lines
- Clean version: ~800 lines (30% reduction)

#### Better Organization

- Audio concerns separated into `AudioConverter`
- Music theory separated into utility functions
- Main generator focused on core generation logic

#### Improved Maintainability

- Clear separation of concerns
- Easier to test individual components
- Reduced complexity in main generator class

#### Better Reusability

- Audio utilities can be used by other parts of the app
- Music theory utilities are pure functions, easy to test

### 4. Migration Strategy

1. **Phase 1**: Update imports in routes to use new utility modules
2. **Phase 2**: Replace `generator.py` with `generator_clean.py`
3. **Phase 3**: Remove old `generator.py` after testing
4. **Phase 4**: Update any remaining imports

### 5. Testing Required

After refactoring, test these endpoints:

- `/api/backing/chords` - Backing track generation
- `/api/seed` - Seed melody generation
- `/api/lead/{id}/preview-seed` - Seed preview
- `/api/lead/{id}` - Lead melody generation
- `/api/lead/{id}/custom` - Custom seed lead melody

## Conclusion

The refactoring will result in:

- **30% smaller main generator file**
- **Better code organization** with separated concerns
- **Improved maintainability** with focused modules
- **No loss of functionality** - all used methods preserved
- **Better testability** with pure utility functions

The refactoring is safe because it only removes unused methods and reorganizes existing functionality without changing the public API.
