# Claude Code Reference for MacLocal API

## Apple Foundation Models Documentation

### GenerationOptions API Reference
- [Apple Developer Documentation - GenerationOptions](https://developer.apple.com/documentation/foundationmodels/generationoptions)

This documentation covers:
- Temperature parameter (0.0-1.0 range for controlling response creativity)
- Sampling options (greedy vs random sampling methods)
- Maximum response tokens configuration
- Usage examples and best practices

### Future Implementation Roadmap

The following advanced sampling parameters will be implemented in future versions:

#### Top-P Sampling (Nucleus Sampling)
- [random(probabilityThreshold:seed:)](https://developer.apple.com/documentation/foundationmodels/generationoptions/samplingmode/random(probabilitythreshold:seed:))
- Controls diversity by only considering tokens whose cumulative probability is below the threshold
- Planned parameter: `--top-p <value>` (0.0-1.0 range)
- More dynamic than top-k as it adapts to the confidence distribution

#### Top-K Sampling
- [random(top:seed:)](https://developer.apple.com/documentation/foundationmodels/generationoptions/samplingmode/random(top:seed:))
- Limits token selection to the K most likely tokens
- Planned parameter: `--top-k <value>` (integer value)
- Provides consistent limitation regardless of probability distribution

These will extend the current `--randomness` parameter to provide more granular control over the sampling behavior.

### Implementation Notes
- Temperature: Controls randomness/creativity in responses (0.0 = deterministic, 1.0 = highly creative)
- Randomness: "greedy" for deterministic output, "random" for varied output
- Apple defaults are used when parameters are not specified
- All parameters are optional and validated at CLI parsing level

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

# Test with both parameters
./afm -t 0.5 -r greedy -s "test prompt"

# Test validation (should fail)
./afm -t 1.5 -s "test prompt"  # Temperature out of range
./afm -r invalid -s "test prompt"  # Invalid randomness value

# Test with debug logging
AFM_DEBUG=1 ./afm -t 1.0 -r greedy -s "test prompt"
```