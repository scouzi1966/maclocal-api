#include "include/cxgrammar/error_handler.h"
#include "include/cxgrammar/tokenizer_info.h"
#include <xgrammar/xgrammar.h>
#include <vector>
#include <string>

using namespace xgrammar;

extern "C" {

void* tokenizer_info_new(const char* const* vocab, size_t vocab_size,
                         const int vocab_type,
                         const int32_t* eos_tokens, size_t eos_tokens_size) {
    try {
        std::vector<std::string> vocab_vec(vocab, vocab + vocab_size);
        std::vector<int32_t> eos_vec(eos_tokens, eos_tokens + eos_tokens_size);
        auto* info = new TokenizerInfo(
            vocab_vec,
            static_cast<VocabType>(vocab_type),
            static_cast<int>(vocab_size),
            eos_vec
        );
        return static_cast<void*>(info);
    } catch (const std::exception& e) {
        catch_error(e.what());
        return nullptr;
    }
}

void tokenizer_info_free(void* tokenizer_info) {
    delete static_cast<TokenizerInfo*>(tokenizer_info);
}

}
