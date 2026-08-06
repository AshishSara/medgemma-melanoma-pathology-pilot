# V3 formal outcome: incomplete

V3 did **not** pass the formal gate and must not be resumed or reported as a
successful pilot.

The frozen run completed three of twenty held-out inputs. It stopped on
`MEL-104-B` under `ocr_degraded` during strict validation of the blind-audit
JSON. The append-only failure event records:

- stage: `audit_validation`;
- error: `StrictGeneratedJSONError`;
- one valid model response for the current row;
- `resume_permitted_for_current_row: false`; and
- three completed rows before the failure.

The incomplete evaluator wrote `results/v3/formal/metrics.json` with
`ground_truth_read: false` and `pass: false`. No v3 accuracy metric was
calculated.

## Root cause

Post-failure diagnostics used only retained model outputs and the pinned
tokenizer, not formal ground truth:

- successful candidate responses used 225–275 of 512 allowed tokens;
- successful blind-audit responses used 656–766 of 768 allowed tokens; and
- the failed audit reached the 768-token boundary before producing complete
  JSON.

Inspection of LM Format Enforcer 0.11.3 also showed that
`TokenEnforcer.__init__` replaces the supplied parser configuration with a
tokenizer-alphabet configuration. Consequently, v3's requested
`force_json_field_order=true` setting was not effective, and the default
twelve consecutive whitespace characters remained allowed. LM Format
Enforcer constrained the generated prefix, but the token cap could still end
generation before the prefix became a complete JSON document.

The v3 lock, artifacts, and failure ledger remain immutable. Any recovery must
use a new protocol version and must not retry or relabel the v3 attempt.
