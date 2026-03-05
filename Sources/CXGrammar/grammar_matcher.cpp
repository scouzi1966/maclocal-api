#include "include/cxgrammar/error_handler.h"
#include "include/cxgrammar/grammar_matcher.h"
#include <xgrammar/xgrammar.h>
#include <dlpack/dlpack.h>

using namespace xgrammar;

extern "C" {

void* grammar_matcher_new(void* compiled_grammar) {
    try {
        auto* cg = static_cast<CompiledGrammar*>(compiled_grammar);
        return new GrammarMatcher(*cg);
    } catch (const std::exception& e) {
        catch_error(e.what());
        return nullptr;
    }
}

bool grammar_matcher_get_bitmask(void* matcher, int32_t* bitmask_out, int bitmask_size) {
    try {
        auto* m = static_cast<GrammarMatcher*>(matcher);

        // Build a DLTensor wrapping the caller's buffer
        DLTensor tensor;
        tensor.data = bitmask_out;
        tensor.ndim = 1;
        tensor.dtype = {kDLInt, 32, 1};
        DLDevice dev = {kDLCPU, 0};
        tensor.device = dev;
        int64_t shape = static_cast<int64_t>(bitmask_size);
        tensor.shape = &shape;
        tensor.strides = nullptr;
        tensor.byte_offset = 0;

        return m->FillNextTokenBitmask(&tensor);
    } catch (const std::exception& e) {
        catch_error(e.what());
        return false;
    }
}

bool grammar_matcher_accept_token(void* matcher, int32_t token_id) {
    try {
        auto* m = static_cast<GrammarMatcher*>(matcher);
        return m->AcceptToken(token_id);
    } catch (const std::exception& e) {
        catch_error(e.what());
        return false;
    }
}

bool grammar_matcher_is_terminated(void* matcher) {
    try {
        auto* m = static_cast<GrammarMatcher*>(matcher);
        return m->IsTerminated();
    } catch (const std::exception& e) {
        catch_error(e.what());
        return false;
    }
}

void grammar_matcher_reset(void* matcher) {
    try {
        auto* m = static_cast<GrammarMatcher*>(matcher);
        m->Reset();
    } catch (const std::exception& e) {
        catch_error(e.what());
    }
}

void grammar_matcher_free(void* matcher) {
    delete static_cast<GrammarMatcher*>(matcher);
}

int grammar_bitmask_size(int vocab_size) {
    return xgrammar::GetBitmaskSize(vocab_size);
}

}
