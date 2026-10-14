# Portability validation

The repository separates machine configuration from pinned sources, packages, and model assets. The completed two-step training result is described in the [smoke summary](../experiments/pilot-smoke-v2/summary.json); the following checks validate the later portable wrappers without running longer training.

## Checks on the existing host

The system Python entrypoints were invoked from outside the checkout. They selected the configured environment and performed these real checks against the existing pilot and checkpoint:

- Validated all 2,116 state/action rows and decoded all 2,116 video frames across five episodes. Float32 state/actions and instruction text matched the frozen raw snapshot exactly. Video-to-source comparisons used sampled frames because encoding is lossy.
- Loaded 15 observation/action samples through the pinned GR00T loader: state `(1, 26)`, action `(16, 26)`, image `(1, 480, 640, 3)`, with matching language.
- Reloaded the checkpoint in a fresh process with pinned offline model assets and produced finite float32 absolute actions `(16, 26)`. Predictions matched the historical smoke reload exactly, with maximum absolute difference 0.0.
- Passed 22 focused preparation/portability tests, including a relocated checkout with spaces, an imported-helper subprocess re-exec regression, local archive hashes and no-overwrite checks, and the parameterized SSH verifier on small local fixtures. Nine existing command-line help entrypoints also passed from outside the checkout. Direct invocation with the configured GR00T interpreter and no preloaded native-library environment also passed all 15 loader samples; the wrapper restarted the actual caller once with the required runtime environment.

Machine-local report files remain ignored. Their content identities are recorded below; they identify evidence on the originating machine and do not imply those reports are distributed here.

| Report basename | SHA-256 |
| --- | --- |
| `portability-reload.json` | `3c87193750a1bd61b4e8e6c8bfc2279e42825c4586cde5221f8622ce7cc2c58b` |
| `portability-loader-validation.json` | `f3c0180cca11db1bb20f6de59356c2b7e77cdb5df271761f7f38f744d58a77a0` |
| `portability-data-validation.json` | `07ae45ba2140d1444d454ea1ed8923f621ef0d6053a297a2e921b2a2bc2262d9` |
| `portability-direct-loader-validation.json` | `30d0a4014498e0f8fee72f89b8c1fb8625b788b563bfb44e8bf4086dfa81d00a` |

## Relocated checkout verification

A separate checkout was verified from the staged repository tree. All recursive source pins were initialized, using local source references to avoid duplicate downloads. Identical immutable Git packs are shared by hardlink. Only unused LIBERO simulator assets and ManiSkill2 simulator data were sparse-omitted in that verification copy; the original source checkouts were preserved.

All four new environments were built in the relocated checkout. Exact installed package versions and editable-source paths matched the locks and relocated source trees. The 19 readiness checks passed, including actual dataset loading, encoded-video decoding, selected-device VRAM and host/output capacity, CUDA backward computation, and FlashAttention backward computation.

A repeated all-stage setup passed without changing environment or distribution metadata timestamps. The bootstrap plan also ran under `/usr/bin/python3 -B -S`, without site packages. Bootstrap `uv` was downloaded independently and its pinned SHA-256 checked.

Pinned package and model caches were shared to preserve disk space. Existing Python 3.10.12/3.12.2 installations and the verified FFmpeg prefix were reused; new environments pointed to the relocated source checkouts. Every required model file was SHA-256 checked. An isolated gated-download test fetched the 1,505-byte Cosmos `config.json` at the pinned revision and matched its expected hash; it did not download model weights. Credentials remained private and were not recorded. More than 8 GiB remained free on the project filesystem.

All 36 tests passed: 14 bootstrap/readiness tests and 22 preparation/portability tests. The verifier did not start training. The following machine-local report hashes identify this validation:


| Report basename | SHA-256 |
| --- | --- |
| `clean-checkout-readiness.json` | `cee0ba87ec68e57599819737004b831a1d73a3c597fc85d2f47e9cb3e507ed5e` |
| `setup-idempotency.json` | `660c9265ddb2a29ebdd457380e26614b8e08747b826068c9e20f6c0b5db2e1c4` |
| `gated-download-test.json` | `5d86f38035a9ff5cd766b80077b52082acac24fc1e4cd4d76c4ab302e65b6335` |
| `repository-bootstrap-validation.json` | `16e08c62d3fea2a07ee6f6e0e01aa5d0f62e734812feece2a9366f5eccab7966` |

## Remaining limits

A completely fresh operating system, empty package caches, fresh gated-model authorization, and complete network downloads have not been tested end to end. Ubuntu 22.04 x86-64 is the supported bootstrap target; other platforms are not claimed. Cached relocation verifies path handling and pinned dependency reconstruction but cannot prove all future network services or system installations.

No new optimizer updates, longer fitting experiment, held-out evaluation, or robot commands were part of portability validation. The 512-step experiment remains proposed.
