#include "include/cxgrammar/error_handler.h"
#include "include/cxgrammar/grammar_compiler.h"
#include <xgrammar/xgrammar.h>
#include <string>
#include <optional>

using namespace xgrammar;

extern "C" {

void* compile_json_schema_grammar(void* tokenizer_info,
                                  const char* schema, size_t length,
                                  int indent) {
    try {
        auto* info = static_cast<TokenizerInfo*>(tokenizer_info);
        GrammarCompiler compiler(*info);
        std::string schema_str(schema, length);
        std::optional<int> indent_opt = indent >= 0 ? std::optional<int>(indent) : std::nullopt;
        auto compiled = compiler.CompileJSONSchema(schema_str, true, indent_opt);
        return new CompiledGrammar(compiled);
    } catch (const std::exception& e) {
        catch_error(e.what());
        return nullptr;
    }
}

void* compile_ebnf_grammar(void* tokenizer_info,
                            const char* grammar_str, size_t length) {
    try {
        auto* info = static_cast<TokenizerInfo*>(tokenizer_info);
        GrammarCompiler compiler(*info);
        std::string ebnf(grammar_str, length);
        auto grammar = Grammar::FromEBNF(ebnf);
        auto compiled = compiler.CompileGrammar(grammar);
        return new CompiledGrammar(compiled);
    } catch (const std::exception& e) {
        catch_error(e.what());
        return nullptr;
    }
}

void* compile_regex_grammar(void* tokenizer_info,
                             const char* regex, size_t length) {
    try {
        auto* info = static_cast<TokenizerInfo*>(tokenizer_info);
        GrammarCompiler compiler(*info);
        std::string regex_str(regex, length);
        auto compiled = compiler.CompileRegex(regex_str);
        return new CompiledGrammar(compiled);
    } catch (const std::exception& e) {
        catch_error(e.what());
        return nullptr;
    }
}

void compiled_grammar_free(void* compiled_grammar) {
    delete static_cast<CompiledGrammar*>(compiled_grammar);
}

}
