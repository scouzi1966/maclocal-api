#pragma once
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void* grammar_matcher_new(void* compiled_grammar);
bool  grammar_matcher_accept_token(void* matcher, int32_t token_id);
bool  grammar_matcher_is_terminated(void* matcher);
void  grammar_matcher_reset(void* matcher);
void  grammar_matcher_free(void* matcher);

/// Returns the number of int32 elements needed for a bitmask covering vocab_size tokens.
int grammar_bitmask_size(int vocab_size);

/// Fill bitmask with allowed tokens. `bitmask_out` must be pre-allocated with
/// grammar_bitmask_size() int32 elements. Returns true if mask needs applying.
bool grammar_matcher_get_bitmask(void* matcher, int32_t* bitmask_out, int bitmask_size);

#ifdef __cplusplus
}
#endif
