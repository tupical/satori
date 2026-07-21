use serde::Deserialize;
use serde_json::{json, Value};

use crate::{
    wrap_untrusted, AiOutput, AiProvider, AiRequest, AiUsage, Confidence, SensemakingError,
    SensingItem, SensingItemKind,
};

#[derive(Deserialize)]
struct SenseResult {
    kind: SensingItemKind,
    confidence: f32,
    summary: String,
}

/// Classify and summarize raw material into the existing sensing artifact.
pub async fn sense_ai<P: AiProvider>(
    provider: &P,
    material: &str,
) -> Result<(SensingItem, Option<AiUsage>), SensemakingError> {
    let req = AiRequest {
        input: Value::String(format!(
            "Classify the material and return a concise factual summary.\n{}",
            wrap_untrusted("material", material)
        )),
        tools: vec![json!({
            "type": "function",
            "name": "sense_material",
            "description": "Return a structured sensing item.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["knowledge", "question", "hypothesis", "risk", "contradiction", "insight", "rejected_idea", "research_gap"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "summary": {"type": "string"}
                },
                "required": ["kind", "confidence", "summary"],
                "additionalProperties": false
            }
        })],
        tool_choice: Some("required".into()),
    };
    let (outputs, usage) = provider.respond_with_usage(req).await?;
    let call = outputs
        .into_iter()
        .find_map(|output| match output {
            AiOutput::ToolCall(call) if call.name == "sense_material" => Some(call),
            _ => None,
        })
        .ok_or_else(|| SensemakingError::ai("sense_ai: model returned no sense_material call"))?;
    let result: SenseResult = serde_json::from_str(&call.arguments)
        .map_err(|e| SensemakingError::serde(e.to_string()))?;
    if !(0.0..=1.0).contains(&result.confidence) || result.summary.trim().is_empty() {
        return Err(SensemakingError::validation(
            "sense_ai: confidence must be 0..=1 and summary must be non-empty",
        ));
    }
    Ok((
        SensingItem::new(result.kind, result.summary)
            .with_confidence(Confidence::new(result.confidence)),
        usage,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AiError, ToolCall};

    struct Fake(Result<Vec<AiOutput>, AiError>);

    impl AiProvider for Fake {
        async fn respond(&self, _req: AiRequest) -> Result<Vec<AiOutput>, AiError> {
            self.0.clone()
        }
    }

    #[tokio::test]
    async fn maps_structured_tool_call() {
        let fake = Fake(Ok(vec![AiOutput::ToolCall(ToolCall {
            name: "sense_material".into(),
            arguments: r#"{"kind":"risk","confidence":0.8,"summary":"Token may expire."}"#.into(),
        })]));
        let (item, _) = sense_ai(&fake, "long material").await.unwrap();
        assert_eq!(item.kind, SensingItemKind::Risk);
        assert_eq!(item.body, "Token may expire.");
        assert_eq!(item.confidence.value(), 0.8);
    }

    #[tokio::test]
    async fn propagates_provider_error() {
        let error = sense_ai(&Fake(Err(AiError::new("boom"))), "material")
            .await
            .unwrap_err();
        assert!(error.to_string().contains("boom"));
    }
}
