#ifndef AFM_C_DWARFSTAR_H
#define AFM_C_DWARFSTAR_H

#include "../../../vendor/ds4/ds4.h"

int afm_ds4_engine_open(
    ds4_engine **out,
    const char *model_path,
    int context_size,
    uint32_t prefill_chunk,
    int power_percent,
    const char *dspark_support_path,
    int dspark_draft_tokens,
    float dspark_confidence_threshold,
    int dspark_strict,
    const char *metal_source_root,
    char *error,
    size_t error_capacity
);

void afm_ds4_tokens_init(ds4_tokens *tokens);
void afm_ds4_free(void *pointer);

typedef struct afm_ds4_prefix_cache afm_ds4_prefix_cache;

int afm_ds4_prefix_cache_open(
    afm_ds4_prefix_cache **out,
    const char *directory,
    uint64_t budget_mb,
    char *error,
    size_t error_capacity
);
void afm_ds4_prefix_cache_close(afm_ds4_prefix_cache *cache);
int afm_ds4_prefix_cache_restore(
    afm_ds4_prefix_cache *cache,
    ds4_engine *engine,
    ds4_session *session,
    ds4_tokens *prompt,
    char *error,
    size_t error_capacity
);
int afm_ds4_prefix_cache_store_session(
    afm_ds4_prefix_cache *cache,
    ds4_engine *engine,
    ds4_session *session,
    const char *reason,
    char *error,
    size_t error_capacity
);

#endif
