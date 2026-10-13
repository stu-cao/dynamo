// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use dynamo_renderer::{OAIPromptFormatter, deepseek::v41::DeepSeekV41Formatter};
use serde_json::{Value, json};

use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::model_type::ModelType;
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_llm::protocols::common::preprocessor::{MultimodalData, PreprocessedRequest};
use dynamo_llm::protocols::openai::responses::NvCreateResponse;

const IMAGE: &str = "<｜deepseek_image｜>";
// One pixel PNG; no network or model weights are needed by these tests.
const INLINE_IMAGE: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a/1cAAAAASUVORK5CYII=";
// The fixture's BPE vocabulary is empty; HF assigns its added tokens 0..N.
const IMAGE_TOKEN: u32 = 2;

fn image_preprocessor() -> (tempfile::TempDir, std::sync::Arc<OpenAIPreprocessor>) {
    let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
    let dir = tempfile::tempdir().unwrap();
    for file in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
    ] {
        std::fs::copy(source.join(file), dir.path().join(file)).unwrap();
    }
    let config_path = dir.path().join("config.json");
    let mut config: Value = serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
    config["model_type"] = json!("deepseek_v41");
    config["image_token_id"] = json!(IMAGE_TOKEN);
    std::fs::write(config_path, serde_json::to_vec(&config).unwrap()).unwrap();
    // Reuse a reserved token in the small local tokenizer to exercise an actual
    // image special token without downloading the DeepSeek tokenizer or weights.
    let tokenizer_path = dir.path().join("tokenizer.json");
    let mut tokenizer: Value =
        serde_json::from_slice(&std::fs::read(&tokenizer_path).unwrap()).unwrap();
    let image_token = tokenizer["added_tokens"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|token| token["content"] == "<|reserved_special_token_0|>")
        .unwrap();
    image_token["content"] = json!(IMAGE);
    std::fs::write(tokenizer_path, serde_json::to_vec(&tokenizer).unwrap()).unwrap();
    let mut mdc = ModelDeploymentCard::load_from_disk(dir.path(), None).unwrap();
    mdc.model_type = ModelType::Chat;
    (dir, OpenAIPreprocessor::new(mdc).unwrap())
}

fn image_urls(request: &PreprocessedRequest) -> Vec<&str> {
    request.multi_modal_data.as_ref().unwrap()["image_url"]
        .iter()
        .map(|image| match image {
            MultimodalData::Url(url) => url.as_str(),
            other => panic!("expected URL passthrough, got {other:?}"),
        })
        .collect()
}

fn image_count(request: &PreprocessedRequest) -> usize {
    request
        .token_ids
        .iter()
        .filter(|&&id| id == IMAGE_TOKEN)
        .count()
}

fn render(fields: Value) -> anyhow::Result<String> {
    let mut payload =
        json!({"model":"deepseek-v4.1", "messages":[{"role":"user", "content":"Hello"}]});
    payload
        .as_object_mut()
        .unwrap()
        .extend(fields.as_object().unwrap().clone());
    let mut request: NvCreateChatCompletionRequest = serde_json::from_value(payload)?;
    request.normalize_reasoning_template_args()?;
    DeepSeekV41Formatter.render(&request)
}

#[test]
fn normalized_effort_matches_the_reference_encoder_for_both_request_fields() {
    assert!(render(json!({})).unwrap().contains("Reasoning Effort: 75 "));
    assert!(render(json!({"reasoning_effort":"xhigh"})).is_err());
    for field in ["chat_template_args", "chat_template_kwargs"] {
        for (effort, budget) in [("low", 50), ("high", 75), ("max", 100)] {
            for fields in [
                json!({"reasoning_effort":effort}),
                json!({(field):{"reasoning_effort":effort}}),
            ] {
                let output = render(fields).unwrap();
                assert!(output.contains(&format!("Reasoning Effort: {budget} ")));
                assert!(output.ends_with("<think>"));
            }
        }
        assert!(
            render(json!({(field):{"reasoning_effort":37}}))
                .unwrap()
                .contains("Reasoning Effort: 37 ")
        );
        assert!(
            render(json!({"reasoning_effort":"low",(field):{"reasoning_effort":37}}))
                .unwrap()
                .contains("Reasoning Effort: 50 ")
        );
        for fields in [
            json!({"reasoning_effort":"none"}),
            json!({(field):{"reasoning_effort":"none"}}),
            json!({(field):{"thinking":false}}),
        ] {
            let output = render(fields).unwrap();
            assert!(!output.contains("Reasoning Effort:"));
            assert!(output.ends_with("</think>"));
        }
        for invalid in [
            json!(0),
            json!(101),
            json!(true),
            json!(1.5),
            json!("xhigh"),
        ] {
            assert!(render(json!({(field):{"reasoning_effort":invalid}})).is_err());
        }
    }
}

#[tokio::test]
async fn chat_images_reach_worker_with_placeholders_and_original_payloads() {
    let (_dir, preprocessor) = image_preprocessor();
    for effort in ["none", "high"] {
        let mut request: NvCreateChatCompletionRequest = serde_json::from_value(json!({
            "model":"deepseek-v4.1", "stream":true, "max_tokens":32,
            "reasoning_effort":effort,
            "messages":[{"role":"user","content":[
                {"type":"text","text":"Compare the images."},
                {"type":"image_url","image_url":{"url":"https://example.com/image.jpg"},"uuid":"remote"},
                {"type":"text","text":"Second image:"},
                {"type":"image_url","image_url":{"url":INLINE_IMAGE},"uuid":"inline"}
            ]}]
        })).unwrap();
        request.normalize_reasoning_template_args().unwrap();
        let original_messages = serde_json::to_value(&request.inner.messages).unwrap();
        let (result, _, thinking) = preprocessor
            .preprocess_request(&request, None)
            .await
            .unwrap();
        assert_eq!(thinking, effort != "none");
        assert_eq!(image_count(&result), 2);
        assert_eq!(
            image_urls(&result),
            ["https://example.com/image.jpg", INLINE_IMAGE]
        );
        assert_eq!(
            result.multi_modal_uuids.as_ref().unwrap()["image_url"],
            [Some("remote".into()), Some("inline".into())]
        );
        let extra = result.extra_args.as_ref().unwrap();
        assert_eq!(extra["messages"], original_messages);
        assert!(
            extra["formatted_prompt"]
                .as_str()
                .unwrap()
                .contains(&format!(
                    "Compare the images.\n\n{IMAGE}\n\nSecond image:\n\n{IMAGE}"
                ))
        );
        assert_eq!(
            serde_json::to_value(&request.inner.messages).unwrap(),
            original_messages
        );
    }
}

#[tokio::test]
async fn responses_input_image_reaches_native_renderer_and_worker() {
    let (_dir, preprocessor) = image_preprocessor();
    let response: NvCreateResponse = serde_json::from_value(json!({
        "model":"deepseek-v4.1", "max_output_tokens":32,
        "chat_template_kwargs":{"reasoning_effort":"high"},
        "input":[{"role":"user","content":[
            {"type":"input_text","text":"What do you see in this image?"},
            {"type":"input_image","image_url":INLINE_IMAGE,"detail":"auto"}
        ]}]
    }))
    .unwrap();
    let mut request = NvCreateChatCompletionRequest::try_from(response).unwrap();
    request.normalize_reasoning_template_args().unwrap();
    let (result, _, thinking) = preprocessor
        .preprocess_request(&request, None)
        .await
        .unwrap();
    assert!(thinking);
    assert_eq!(image_count(&result), 1);
    assert_eq!(image_urls(&result), [INLINE_IMAGE]);
    assert!(
        result.extra_args.as_ref().unwrap()["formatted_prompt"]
            .as_str()
            .unwrap()
            .ends_with(&format!(
                "What do you see in this image?\n\n{IMAGE}<｜Assistant｜><think>"
            ))
    );
}

#[tokio::test]
async fn tool_images_and_cache_ids_follow_rendered_call_order() {
    let (_dir, preprocessor) = image_preprocessor();
    let request: NvCreateChatCompletionRequest = serde_json::from_value(json!({
        "model":"deepseek-v4.1", "max_tokens":32,
        "messages":[
            {"role":"assistant","tool_calls":[
                {"id":"a","type":"function","function":{"name":"screenshot","arguments":"{}"}},
                {"id":"b","type":"function","function":{"name":"screenshot","arguments":"{}"}}
            ]},
            {"role":"tool","tool_call_id":"b","content":[
                {"type":"text","text":"SECOND_RESULT"},
                {"type":"image_url","image_url":{"url":"https://example.com/b.png"},"uuid":"b"},
                {"type":"image_url","uuid":"b-cached"}
            ]},
            {"role":"user","content":[
                {"type":"text","text":"USER_IMAGE"},
                {"type":"image_url","image_url":{"url":INLINE_IMAGE},"uuid":"user"}
            ]},
            {"role":"tool","tool_call_id":"a","content":[
                {"type":"text","text":"FIRST_RESULT"},
                {"type":"image_url","image_url":{"url":"https://example.com/a.png"},"uuid":"a"}
            ]}
        ]
    }))
    .unwrap();
    let (result, _, _) = preprocessor
        .preprocess_request(&request, None)
        .await
        .unwrap();
    assert_eq!(image_count(&result), 4);
    let images = &result.multi_modal_data.as_ref().unwrap()["image_url"];
    let identities: Vec<_> = images
        .iter()
        .map(|image| match image {
            MultimodalData::Url(url) => url.as_str(),
            MultimodalData::UuidOnly(uuid) => uuid.as_str(),
            other => panic!("unexpected image: {other:?}"),
        })
        .collect();
    assert_eq!(
        identities,
        [
            "https://example.com/a.png",
            INLINE_IMAGE,
            "https://example.com/b.png",
            "b-cached"
        ]
    );
    assert_eq!(
        result.multi_modal_uuids.as_ref().unwrap()["image_url"],
        [
            Some("a".into()),
            Some("user".into()),
            Some("b".into()),
            Some("b-cached".into())
        ]
    );
    let prompt = result.extra_args.as_ref().unwrap()["formatted_prompt"]
        .as_str()
        .unwrap();
    assert!(prompt.find("FIRST_RESULT").unwrap() < prompt.find("USER_IMAGE").unwrap());
    assert!(prompt.find("USER_IMAGE").unwrap() < prompt.find("SECOND_RESULT").unwrap());
    // Original messages stay available to workers in API order.
    assert_eq!(
        result.extra_args.as_ref().unwrap()["messages"],
        serde_json::to_value(&request.inner.messages).unwrap()
    );
}

#[test]
fn continuing_a_reordered_final_tool_result_is_rejected() {
    let (_dir, preprocessor) = image_preprocessor();
    let request: NvCreateChatCompletionRequest = serde_json::from_value(json!({
        "model":"deepseek-v4.1", "continue_final_message":true,
        "add_generation_prompt":false,
        "messages":[
            {"role":"assistant","tool_calls":[
                {"id":"a","type":"function","function":{"name":"screenshot","arguments":"{}"}},
                {"id":"b","type":"function","function":{"name":"screenshot","arguments":"{}"}}
            ]},
            {"role":"tool","tool_call_id":"b","content":"second"},
            {"role":"tool","tool_call_id":"a","content":"first"}
        ]
    }))
    .unwrap();
    assert!(
        preprocessor
            .apply_template(&request)
            .unwrap_err()
            .to_string()
            .contains("prompt formatter moves it")
    );
}
