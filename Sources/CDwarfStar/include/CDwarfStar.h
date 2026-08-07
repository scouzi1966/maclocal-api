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

#endif
