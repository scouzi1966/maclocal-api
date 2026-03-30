# Model Path Resolution

How AFM resolves model locations for MLX inference. Covers environment variables, lookup order, download behavior, and the upstream swift-transformers quirk that makes `MACAFM_MLX_MODEL_CACHE` important.

## Environment Variables

| Variable | Who sets it | Purpose |
|---|---|---|
| `MACAFM_MLX_MODEL_CACHE` | User | Primary cache root. Controls both lookup (candidate #1) and download destination. |
| `HF_HOME` | `applyEnvironment()` (from `MACAFM_MLX_MODEL_CACHE`) | Read by our resolver for lookup (candidate #4). Ignored by swift-transformers for downloads. |
| `HUGGINGFACE_HUB_CACHE` | `applyEnvironment()` (from `MACAFM_MLX_MODEL_CACHE`) | Read by our resolver for lookup (candidate #3). Ignored by swift-transformers for downloads. |
| `HF_HUB_CACHE` | User / external | Read by our resolver for lookup (candidate #3). Ignored by swift-transformers for downloads. |
| `XDG_CACHE_HOME` | User / OS | Read by our resolver for lookup (candidate #5). Ignored by swift-transformers for downloads. |

**Key point:** swift-transformers' `HubApi` does not read any environment variable for its download directory. It only accepts `downloadBase` as an explicit constructor parameter.

## Lookup Order (MLXCacheResolver.localModelDirectory)

When AFM needs to find an existing model on disk, it checks these paths in order. The first match with valid files (`config.json` + `*.safetensors`) wins.

For a model like `mlx-community/Qwen3.5-35B-A3B-4bit`:

### 1. MACAFM_MLX_MODEL_CACHE (if set)

Three sub-paths checked:

```
{MACAFM_MLX_MODEL_CACHE}/mlx-community/Qwen3.5-35B-A3B-4bit          # flat
{MACAFM_MLX_MODEL_CACHE}/models/mlx-community/Qwen3.5-35B-A3B-4bit   # models subdir
{MACAFM_MLX_MODEL_CACHE}/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit  # HF-style
```

### 2. Swift Hub default

```
~/Documents/huggingface/models/mlx-community/Qwen3.5-35B-A3B-4bit
```

This is where `HubApi.shared` downloads to by default (hardcoded in swift-transformers).

### 3. HUGGINGFACE_HUB_CACHE / HF_HUB_CACHE (if set)

```
{HUGGINGFACE_HUB_CACHE}/models--mlx-community--Qwen3.5-35B-A3B-4bit
```

### 4. HF_HOME (if set)

```
{HF_HOME}/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit
```

### 5. XDG_CACHE_HOME (if set)

```
{XDG_CACHE_HOME}/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit
```

### 6. Default Python HF cache

```
~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit
```

### 7. macOS Library/Caches (legacy)

```
~/Library/Caches/models/mlx-community/Qwen3.5-35B-A3B-4bit
~/Library/Caches/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit
```

### Snapshot resolution

For HF-style directories with a `snapshots/` subdirectory, the resolver enters the first snapshot hash directory and checks for required files there.

## Download Destination

When a model is not found locally, `MLXModelService.downloadModel()` calls `HubApi.snapshot()`:

- **`MACAFM_MLX_MODEL_CACHE` is set:** Creates `HubApi(downloadBase: cacheRoot)`. Downloads go to `{MACAFM_MLX_MODEL_CACHE}/models/{org}/{model}`.
- **`MACAFM_MLX_MODEL_CACHE` is NOT set:** Creates `HubApi()` with no parameter. Downloads go to `~/Documents/huggingface/models/{org}/{model}`.

## The ~/Documents/huggingface Problem

swift-transformers (`huggingface/swift-transformers`) hardcodes `downloadBase` to `~/Documents/huggingface/` when no explicit parameter is passed ([HubApi.swift L120-121](https://github.com/huggingface/swift-transformers/blob/1.3.0/Sources/Hub/HubApi.swift#L120-L121)):

```swift
let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
self.downloadBase = documents.appending(component: "huggingface")
```

This causes two problems on macOS:

1. **iCloud sync:** If `~/Documents` is managed by iCloud Drive, large model files may be evicted from local storage. Downloads can hang or time out when disk space is low.
2. **TCC permission prompts:** Apps accessing `~/Documents` trigger macOS privacy permission dialogs.

Upstream issues:
- [swift-transformers #102](https://github.com/huggingface/swift-transformers/issues/102) — Requested `~/.cache/huggingface/hub` support. Claimed resolved in 1.2.0 via `HubCache`, but `downloadBase` still defaults to `~/Documents/huggingface` in 1.3.0.
- [swift-transformers #339](https://github.com/huggingface/swift-transformers/issues/339) — TCC prompt triggered by the `~/Documents` default.

## Recommendation

Always set `MACAFM_MLX_MODEL_CACHE` to a directory outside iCloud scope:

```bash
export MACAFM_MLX_MODEL_CACHE=~/.cache/afm
```

AFM prints a warning at startup when this variable is not set.
