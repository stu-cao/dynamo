# Prime Intellect dynamo maintenance

This repo carries Prime Intellect fix branches on top of upstream
[ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo). It is a duplicate
mirror, not a GitHub fork, so it can stay private.

## Branch layout

| branch | contents |
|---|---|
| `main` | verbatim mirror of upstream `main` (reference only, never deployed) |
| `prime/nightly-20260910` | maintenance line pinned to upstream nightly `1.5.0.dev20260910` = `287a8e76881aaf433a376796546ca703f09d96ba`, plus Prime fixes below |

The maintenance line is what the inference fleet consumes. Production pins the
official nightly images; a `prime` build is the official nightly **plus exactly
the fixes listed below**.

## Fixes carried on `prime/nightly-20260910`

| commit | fix | upstream PR |
|---|---|---|
| `ec0918bedc` | coalesce duplicate-index entries in streaming chat chunks | [#15054](https://github.com/ai-dynamo/dynamo/pull/15054) (open) |
| `2c8d6649ba` | normalize before glm47 recovery, never fold unmergeable chunks | [#15054](https://github.com/ai-dynamo/dynamo/pull/15054) (open) |
| `84367bc503` | accept `max` reasoning effort on the Responses API | not yet submitted |
| `6161a36c58` | preserve worker errors when local stop conditions trigger | [#15094](https://github.com/ai-dynamo/dynamo/pull/15094) (open) |

Notes:

- The first two commits are hand-ported onto the nightly base; upstream authored
  them against newer `main`, where the glm47 recovery code was refactored
  (`PendingRecovery` -> `ChoiceRecovery`, PR #13792). The ported version keeps
  the nightly's `PendingRecovery` pass and moves the coalesce call before it,
  as the upstream patch comment prescribes.
- When an upstream PR merges, re-pin to a nightly that contains it and drop the
  carried commit (see below).

## Versioning

Release wheels are stamped `<base><suffix>` with the same stamping mechanism
upstream nightly CI uses (`.github/scripts/apply_dev_version.py`). For this
branch the suffix is `.dev20260910+prime.<N>`, e.g. `1.5.0.dev20260910+prime.1`.
The local `+prime.N` segment sorts above the plain nightly version and can never
be confused with an upstream nightly.

## Building release wheels

Actions -> **Prime Wheels** -> Run workflow (on `prime/nightly-20260910`):

- `version_suffix`: stamp for this build, e.g. `.dev20260910+prime.1`.
- `base_ref`: upstream commit the branch is pinned to (release notes).

The workflow builds `ai-dynamo`, `ai-dynamo-runtime`, `gpu-memory-service`,
and stages `aisimulate` for amd64 and arm64 inside the upstream manylinux
`wheel_builder` stage, then publishes everything as a GitHub Release on this
repo (`wheels-*` tags) with SHA256SUMS. Wheels are installable by direct URL
from the release assets.

Everything is compiled inside the upstream container pipeline
(`container/render.py --framework dynamo --target wheel_builder`), so the
wheels match what upstream images embed, byte-for-byte toolchain-wise.

## Adding a fix

1. Cherry-pick the fix onto `prime/nightly-20260910` (keep the original author;
   if it conflicts, port it by hand as the two PR #15054 commits above were).
2. Update the table in this file.
3. Run Prime Wheels with the next `+prime.N` suffix.
4. Push the fix upstream; when it merges, schedule a re-pin.

## Re-pinning to a newer nightly

1. `git fetch origin main` in a local clone (add upstream as `origin`).
2. `git checkout -b prime/nightly-YYYYMMDD <new-nightly-sha>`.
3. Cherry-pick the fixes that are still unmerged upstream; drop the merged ones.
4. Update the table here, push the branch, build wheels.

Keep one branch per nightly the fleet actually runs; delete branches once the
fleet moves on. `main` is refreshed by `git push prime origin/main:refs/heads/main`.
