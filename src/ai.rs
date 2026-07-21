//! Sensemaking's slice of the AI provider seam: re-exports the shared
//! [`layer_kit::ai`] infrastructure and adds the one domain-specific tool
//! schema `research` needs.

use serde_json::{json, Value};

pub use layer_kit::ai::{
    AiError, AiOutput, AiProvider, AiRequest, AiUsage, ToolCall, UNTRUSTED_CLOSE, UNTRUSTED_OPEN,
};
pub use layer_kit::ai::wrap_untrusted;

/// JSON schema for the `research_answer` function tool used by `research`.
pub fn research_answer_tool() -> Value {
    json!({
        "type": "function",
        "name": "research_answer",
        "description": "Return a structured research answer to the posed query.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The answer to the research query, plain text."
                }
            },
            "required": ["answer"]
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn research_answer_tool_shape() {
        let t = research_answer_tool();
        assert_eq!(t["name"], "research_answer");
        assert_eq!(t["parameters"]["required"][0], "answer");
    }
}
