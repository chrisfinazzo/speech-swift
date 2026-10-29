// Adapted from MOSS-VoiceGenerator and MOSS-Audio-Tokenizer.
// Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
// Licensed under Apache 2.0; see LICENSE.
// Native Swift adaptation uses request-local state and bounded causal caches.

import Foundation
import Tokenizers
import Hub

enum MVGPrompt {
    static func render(text:String,instruction:String,normalize:Bool=false)throws->String {
        guard !text.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,
              !instruction.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty else {
            throw MossVoiceGeneratorError.invalidInput("text and voice instruction must both be nonempty")
        }
        // Control tokens belong to the processor, never to user-supplied fields.
        guard !text.contains("<|"),!instruction.contains("<|"),
              !text.contains("</user_inst>"),!instruction.contains("</user_inst>") else {
            throw MossVoiceGeneratorError.invalidInput("input contains reserved processor delimiters")
        }
        let t=normalize ? cleanup(text,instruction:false) : text
        let i=normalize ? cleanup(instruction,instruction:true) : instruction
        guard !t.isEmpty,!i.isEmpty else { throw MossVoiceGeneratorError.invalidInput("normalization removed all input") }
        return "<|im_start|>user\n<user_inst>\n- Reference(s):\nNone\n- Instruction:\n\(i)\n- Tokens:\nNone\n- Quality:\nNone\n- Sound Event:\nNone\n- Ambient Sound:\nNone\n- Language:\nNone\n- Text:\n\(t)\n</user_inst><|im_end|>\n<|im_start|>assistant\n"
    }

    /// Matches the upstream processor, including keeping the last punctuation
    /// character of a repeated punctuation run (Python's repeated capture).
    static func cleanup(_ input:String,instruction:Bool)->String {
        var s=input.replacingOccurrences(of:"\n",with:" ")
        for pattern in ["\\[.*?\\]","\\{.*?\\}"] {
            s=s.replacingOccurrences(of:pattern,with:"",options:.regularExpression)
        }
        for character in "【】《》（）『』「」～"+(instruction ? "-_" : "") {
            s=s.replacingOccurrences(of:String(character),with:"，")
        }
        if !instruction { s=s.replacingOccurrences(of:"\"",with:"") }
        s=s.replacingOccurrences(of:"([，。！？,.!?;；])+",with:"$1",options:.regularExpression)
        if instruction,s.range(of:"[\\u4e00-\\u9fff]",options:.regularExpression) != nil {
            s=s.replacingOccurrences(of:",",with:"，")
        }
        return s.trimmingCharacters(in:.whitespacesAndNewlines)
    }
}

final class MVGTokenizer {
    let tokenizer:Tokenizer
    private init(_ tokenizer:Tokenizer) { self.tokenizer=tokenizer }
    static func load(_ directory:URL)async throws->MVGTokenizer {
        let hub=HubApi()
        let config=try hub.configuration(fileURL:directory.appendingPathComponent("tokenizer_config.json"))
        let data=try hub.configuration(fileURL:directory.appendingPathComponent("tokenizer.json"))
        let value=try AutoTokenizer.from(tokenizerConfig:config,tokenizerData:data,strict:false)
        let expected=["<|im_start|>":151644,"<|im_end|>":151645,"<|audio_start|>":151652,
                      "<|audio_end|>":151653,"<|audio_assistant_gen_slot|>":151656,
                      "<|audio_assistant_delay_slot|>":151662]
        for (token,id) in expected where value.convertTokenToId(token) != id {
            throw MossVoiceGeneratorError.invalidConfiguration("tokenizer control token differs: \(token)")
        }
        return MVGTokenizer(value)
    }
    func encode(_ prompt:String)->[Int] { tokenizer.encode(text:prompt,addSpecialTokens:false) }
}
