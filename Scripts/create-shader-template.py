#!/usr/bin/env python3
"""Create a custom Instruments template with Shader Timeline enabled.

Patches Apple's built-in 'Metal System Trace' template to enable the
shader profiler, which captures per-kernel GPU shader names and timing.
The patched template is saved for use with:
  xcrun xctrace record --template <path> ...
"""
import plistlib, sys, os

SRC = "/Applications/Xcode.app/Contents/Applications/Instruments.app/Contents/Packages/GPU.instrdst/Contents/Templates/Metal System Trace.tracetemplate"
DST = os.path.expanduser("~/Library/Developer/Xcode/UserData/Instruments/Templates/Metal Shader Profile.tracetemplate")

if not os.path.exists(SRC):
    print(f"Error: Source template not found at {SRC}", file=sys.stderr)
    print("Make sure Xcode is installed.", file=sys.stderr)
    sys.exit(1)

with open(SRC, "rb") as f:
    plist = plistlib.load(f)

objects = plist["$objects"]

# Find indices of shader profiler keys
shader_keys = {}
for i, obj in enumerate(objects):
    if obj in ("shaderprofiler", "shaderprofilerinternal"):
        shader_keys[obj] = i

if len(shader_keys) < 2:
    print("Error: Could not find shader profiler keys in template", file=sys.stderr)
    sys.exit(1)

# Find True value in objects array
true_idx = next((i for i, o in enumerate(objects) if o is True), None)
if true_idx is None:
    true_idx = len(objects)
    objects.append(True)

# Find the state dictionary containing these keys
patched = 0
for i, obj in enumerate(objects):
    if not isinstance(obj, dict) or "NS.keys" not in obj:
        continue
    keys = obj["NS.keys"]
    key_uids = [k.data if hasattr(k, "data") else k for k in keys]
    vals = obj["NS.objects"]
    for j, k_uid in enumerate(key_uids):
        if k_uid < len(objects) and objects[k_uid] in shader_keys:
            old_val_uid = vals[j].data if hasattr(vals[j], "data") else vals[j]
            if old_val_uid < len(objects) and objects[old_val_uid] is False:
                vals[j] = plistlib.UID(true_idx)
                patched += 1
    if patched > 0:
        obj["NS.objects"] = vals
        objects[i] = obj
        break

if patched == 0:
    print("Warning: No settings were patched (may already be enabled)", file=sys.stderr)

os.makedirs(os.path.dirname(DST), exist_ok=True)
with open(DST, "wb") as f:
    plistlib.dump(plist, f, fmt=plistlib.FMT_BINARY)

print(f"Patched {patched} shader profiler settings")
print(f"Saved to: {DST}")
