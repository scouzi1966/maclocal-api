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

void grammar_matcher_fill_next_token_bitmask(void* matcher, void* bitmask) {
    try {
        auto* m = static_cast<GrammarMatcher*>(matcher);
        auto* tensor = static_cast<DLTensor*>(bitmask);
        m->FillNextTokenBitmask(tensor);
    } catch (const std::exception& e) {
        catch_error(e.what());
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
