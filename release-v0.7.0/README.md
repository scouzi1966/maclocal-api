# afm v0.7.0 - Apple Foundation Models CLI & API Server

**Enhanced Randomness Parameters with Advanced Sampling Support**

A macOS server and CLI tool that exposes Apple's Foundation Models through OpenAI-compatible API endpoints with advanced sampling control.

## 🚀 What's New in v0.7.0

### **Enhanced Randomness Parameters**
- **Nucleus Sampling (Top-P)**: `random:top-p=<0.0-1.0>` - Controls diversity by probability threshold
- **Top-K Sampling**: `random:top-k=<positive integer>` - Limits selection to K most likely tokens
- **Seeded Random**: `random:seed=<value>` - Reproducible results with specific seeds
- **Combined Modes**: `random:top-p=0.9:seed=42` - Mix parameters for fine control

### **Comprehensive Testing**
- **New Test Suite**: `test-go.sh` with 24+ test cases covering all parameter combinations
- **Flexible Configuration**: Command-line options for binary path and test prompts
- **Complete Validation**: Tests for conflict detection and error handling

### **Enhanced Code Quality**
- **Centralized Debug Logger**: Improved debugging with consistent logging
- **Better Validation**: Clear error messages for invalid parameter combinations
- **Enhanced Documentation**: Comprehensive parameter format reference

## 🎯 Usage Examples

### **CLI Examples**
```bash
# Basic usage (backward compatible)
./afm -r greedy -s "Tell me a story"
./afm -r random -s "Tell me a story"

# Advanced sampling modes
./afm -r "random:top-p=0.9" -s "Tell me a story"           # Nucleus sampling
./afm -r "random:top-k=50" -s "Tell me a story"            # Top-k sampling
./afm -r "random:seed=42" -s "Tell me a story"             # Seeded random
./afm -r "random:top-p=0.9:seed=42" -s "Tell me a story"   # Combined mode

# Temperature with sampling
./afm -t 0.7 -r "random:top-p=0.9" -s "Write a poem"

# Server mode with advanced parameters
./afm -r "random:top-p=0.9:seed=42" -p 9999
```

### **Testing**
```bash
# Run comprehensive test suite
./test-go.sh

# Test with custom binary and prompt
./test-go.sh -b /path/to/afm -s "Custom test prompt"

# Test specific scenarios
./afm -r "random:top-p=0.9:top-k=50" -s "test"  # Should fail with clear error
```

## 📋 Requirements

- **macOS 26+ (Tahoe)** with Apple Intelligence enabled
- **Apple Silicon Mac** (M1/M2/M3/M4 series)
- **Foundation Models framework** available

## 🔧 Key Features

- **🎯 Advanced Sampling Control**: Nucleus, top-k, and seeded random sampling
- **🔒 Privacy-First**: All processing happens locally on your device
- **⚡ Fast & Lightweight**: No network calls, no API keys required
- **🛠️ OpenAI Compatible**: Drop-in replacement for OpenAI API endpoints
- **📊 Comprehensive Testing**: Full validation suite with detailed logging
- **🔄 Backward Compatible**: All existing usage patterns continue to work

## 🚨 Important Notes

### **Parameter Validation**
- **Single Sampling Method**: Cannot combine top-p and top-k (per Apple's API constraints)
- **Valid Ranges**: top-p (0.0-1.0), top-k (positive integers), seeds (non-negative integers)
- **Clear Error Messages**: Helpful validation feedback for invalid combinations

### **Examples of Invalid Usage**
```bash
# These will be rejected with clear error messages
./afm -r "random:top-p=0.9:top-k=50" -s "test"    # Cannot combine sampling methods
./afm -r "random:top-p=1.5" -s "test"             # top-p out of range
./afm -r "random:top-k=-5" -s "test"              # Invalid top-k value
```

## 📖 More Information

- **GitHub Repository**: https://github.com/scouzi1966/maclocal-api
- **Issues & Support**: https://github.com/scouzi1966/maclocal-api/issues
- **Homebrew Formula**: https://github.com/scouzi1966/homebrew-afm

---

**v0.7.0** - Enhanced randomness parameters with advanced sampling support