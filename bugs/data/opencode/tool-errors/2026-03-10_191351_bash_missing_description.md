# Tool Error: bash missing `description` param

## Context
- **Date**: 2026-03-10 19:13:51
- **Model**: mlx-community/Qwen3.5-35B-A3B-4bit
- **afm version**: v0.9.7
- **afm CLI**: `MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache AFM_DEBUG=1 afm mlx -m Qwen3.5-35B-A3B-4bit --enable-prefix-caching --tool-call-parser afm_adaptive_xml -vv --port 9999`
- **OpenCode session**: `ses_325fa4050ffegO1C93oUKQeDUC`
- **Session title**: Simple Local File Organizer PRD
- **afm request-id**: `1C30F6A1-050B-4EBA-85C8-B0A6658B5E2E`

## Classification
**Model generation error** — model emitted `bash(command)` without `description` param. afm correctly parsed what the model sent (1 arg). afm detected the missing required param but policy was to not fill strings (unlike boolean/integer/array/object which get defaults).

## Additional Issue: Thinking Leakage
Same session showed `◁im_start▷assistant` raw tokens leaking into OpenCode's "Thinking:" display. Qwen3.5-35B-A3B supports thinking — this is a separate bug to investigate.

## Raw Data Files
- `2026-03-10_191215_session_afm_full.log` — complete afm server output for the entire session (all requests)
- `2026-03-10_191215_session_all_tool_parts.jsonl` — all 13 tool call parts from OpenCode DB for this session
- `2026-03-10_191351_opencode_raw.json` — the specific error part from OpenCode DB
- `2026-03-10_191351_afm_raw.log` — afm log excerpt for the error request
