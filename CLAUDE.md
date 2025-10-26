# Claude Code Reference for MacLocal API

## Apple Foundation Models Documentation

### GenerationOptions API Reference
- [Apple Developer Documentation - GenerationOptions](https://developer.apple.com/documentation/foundationmodels/generationoptions)

This documentation covers:
- Temperature parameter (0.0-1.0 range for controlling response creativity)
- Sampling options (greedy vs random sampling methods)
- Maximum response tokens configuration
- Usage examples and best practices

### Advanced Sampling Features

The following advanced sampling parameters are now available:

#### Top-P Sampling (Nucleus Sampling)
- [random(probabilityThreshold:seed:)](https://developer.apple.com/documentation/foundationmodels/generationoptions/samplingmode/random(probabilitythreshold:seed:))
- Controls diversity by only considering tokens whose cumulative probability is below the threshold
- Usage: `--randomness "random:top-p=0.9"` (0.0-1.0 range)
- More dynamic than top-k as it adapts to the confidence distribution

#### Top-K Sampling
- [random(top:seed:)](https://developer.apple.com/documentation/foundationmodels/generationoptions/samplingmode/random(top:seed:))
- Limits token selection to the K most likely tokens
- Usage: `--randomness "random:top-k=50"` (integer value)
- Provides consistent limitation regardless of probability distribution

#### Stop Sequences
- Specify strings where the model should stop generating text
- CLI Parameter: `--stop "###,END"` (comma-separated list of stop sequences)
- API Parameter: `"stop": ["###", "END"]` (array of strings)
- When any stop sequence is encountered, generation stops at that point
- The stop sequence itself is excluded from the output
- Multiple stop sequences can be specified
- Works in both streaming and non-streaming modes

These parameters provide granular control over the sampling behavior and output formatting.

### Implementation Notes
- **Temperature**: Controls randomness/creativity in responses (0.0 = deterministic, 1.0 = highly creative)
- **Randomness**: "greedy" for deterministic output, "random" for varied output, or advanced sampling modes
- **Stop Sequences**: Truncate output at specified strings, useful for structured output formats
- Apple defaults are used when parameters are not specified
- All parameters are optional and validated at CLI parsing level
- Stop sequences from CLI and API requests are merged, with duplicates removed

### Build Commands
```bash
# Debug build
swift build

# Release build
swift build --configuration release

# Clean and rebuild
swift package clean && swift build
```

### Debug Logging
Debug logging can be enabled by setting the `AFM_DEBUG` environment variable:

```bash
# Enable debug logging
export AFM_DEBUG=1
./afm -s "test prompt"

# Or inline
AFM_DEBUG=1 ./afm -t 1.0 -r greedy -s "test prompt"

# Disable debug logging
unset AFM_DEBUG
# or
export AFM_DEBUG=0
```

Debug logging shows:
- Parameter values passed to createGenerationOptions
- Foundation Model Service initialization steps
- Response generation progress
- Error details during single-prompt mode

### Testing Parameters
```bash
# Test with temperature only
./afm -t 1.0 -s "test prompt"

# Test with temperature and randomness
./afm -t 0.5 -r greedy -s "test prompt"

# Test with top-p sampling
./afm -r "random:top-p=0.9" -s "test prompt"

# Test with top-k sampling
./afm -r "random:top-k=50" -s "test prompt"

# Test with stop sequences
./afm --stop "###,END" -s "Write a story. ###"

# Test with multiple parameters
./afm -t 0.7 -r "random:top-p=0.95" --stop "---" -s "test prompt"

# Test validation (should fail)
./afm -t 1.5 -s "test prompt"  # Temperature out of range
./afm -r invalid -s "test prompt"  # Invalid randomness value

# Test with debug logging
AFM_DEBUG=1 ./afm -t 1.0 -r greedy --stop "###" -s "test prompt"
```