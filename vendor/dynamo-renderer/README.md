# Renderer

## Introduction
`dynamo-renderer` turns OpenAI-style chat requests into model-ready prompt strings. It is the *encode* side of inference serving: messages + tools + generation settings in, a fully-rendered prompt out. It is standalone, so an external OpenAI frontend can reuse Dynamo's prompt formatting without pulling in the Dynamo runtime.

It renders HuggingFace `chat_template` jinja2 (via `minijinja` + `minijinja-contrib` pycompat) and also ships native Rust formatters for model families whose prompt protocol cannot be represented faithfully by the published template. The crate is a *bridge* between OpenAI request types (`dynamo-protocols`) and the template engine; Kimi K3 additionally returns segment-aware prompts so structural XTML tokens and ordinary user text remain distinct through tokenization.

## Features
- **HF chat templates**: faithful `apply_chat_template` rendering, including tool-use and generation-prompt handling.
- **Native DeepSeek formatters**: Rust formatters for V4.1 / V4 / V3.2 families (under `deepseek`).
- **Native Inkling formatter**: exact text, image, audio, reasoning, and tool-use framing for `inkling_mm_model` (under `inkling`). Media blocks contain only the marker token; the backend multimodal processor expands per-patch or per-frame placeholders later.
- **Native Kimi K3 formatter**: XTML rendering with explicit trusted-control and ordinary-text segment boundaries (under `kimi_k3`).
- **Bring-your-own request type**: implement `OAIChatLikeRequest` for any request type, or use the ready-made impl for `dynamo-protocols`' OpenAI chat request.
- **Self-contained**: no async runtime or networking; segment-aware prompts interoperate with `dynamo-tokenizers`.

## Quick Start

```rust
use dynamo_renderer::{ChatTemplate, ContextMixins, PromptFormatter};
use dynamo_protocols::types::CreateChatCompletionRequest;

// `config` is parsed from a model's `tokenizer_config.json`.
let config: ChatTemplate = serde_json::from_str(tokenizer_config_json)?;
let PromptFormatter::OAI(formatter) =
    PromptFormatter::from_parts(config, ContextMixins::default(), /* exclude_tools_when_tool_choice_none */ false)?
else {
    unreachable!("from_parts always builds an OAI formatter")
};

// Any type implementing `OAIChatLikeRequest` can be rendered; the standard
// OpenAI chat request works out of the box.
let request: CreateChatCompletionRequest = serde_json::from_str(request_json)?;
let prompt: String = formatter.render(&request)?;
```

## Local image-support backport

This copy is based on the published `dynamo-renderer` 5.2.0 crate from
`ai-dynamo/frontend-crates` commit `f04f28c984c4200f51671cadf63e3b53c16c28ea`.
It retains the runtime's `dynamo-protocols` 5.4.3 dependency. The published
renderer 5.4.0 also upgrades protocols to 6.x, so it is not a drop-in update.

The source and reference fixtures backport these upstream changes:

- [#254](https://github.com/ai-dynamo/frontend-crates/pull/254), commit
  `fc601a1f795fd05413f1e114b14e24d124206938`: image placeholders and source validation.
- [#279](https://github.com/ai-dynamo/frontend-crates/pull/279), commit
  `8997f358010b8666834345508a1397e962a65aa1`: shared prompt and media ordering.
- The empty tool-call ordering correction from [#366](https://github.com/ai-dynamo/frontend-crates/pull/366),
  commit `6d025c95fc34fa7fac7b70632907252039d48b41`. Its unrelated changes to the
  shared V4 encoder and raw `task` extension are not included.

`lib/llm/src/preprocessor.rs` consumes the renderer's `media_message_order` hook
when collecting image payloads and UUIDs. Root, Python runtime, and KVBM binding
workspaces all resolve this local crate. Remove the backport when the runtime
upgrades to compatible published frontend crates containing these fixes.

Validate with `cargo test -p dynamo-renderer` and
`cargo test -p dynamo-llm --no-default-features --test deepseek_v41_renderer`.
These checks use local tokenizer fixtures and do not require model weights.

## DeepSeek V4.1

The V4.1 formatter supports text and image messages, tool history, mid-conversation system messages, and numeric reasoning effort. It rejects audio/video content and explicit tool namespace fields; qualified function names are preserved. Generation headers follow the reference encoder and cannot be disabled with `add_generation_prompt`. OpenAI effort names match the reference encoder: `low` is 50, `high` is 75, and `max` is 100; the default is 75. The `xhigh` alias is not supported. Template arguments accept the same names or an integer from 1 to 100. Top-level effort takes precedence over template effort. Set `reasoning_effort` to `none` or the template argument `thinking` to `false` to disable thinking.

Reasoning-effort names and the default follow the model's [Python reference encoder at revision `dba1be0a`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/encoding/encoding.py#L444). Prompt fixtures pass explicit numeric effort to the low-level encoder. Image cases use the same fixture schema and include URL/data-URL inputs, mixed content, multiple turns and tool results. Their expected prompts were generated with the pinned reference encoder (SHA-256 `502bdaec8a3fd88ebc24c4721a7038fbe42f2063c664638127056107920035c1`). UUID-only image reuse is a Dynamo processor-cache extension tested separately.

Prompt-format tests do not replace live image-fetch, transfer, cache and backend validation. The runtime must collect tool-result images in the same call order as their rendered placeholders.

## Relationship to other crates
- `dynamo-protocols` — OpenAI/wire request types this crate renders from.
- `dynamo-tokenizers` — tokenization (the *next* step after rendering); re-exported here for convenience.
- `dynamo-parsers` — the *decode* side (parsing model output back into reasoning / tool calls).

## Ownership

This crate is developed and published from frontend-crates. This runtime temporarily
vendors the compatible backport described above.
