# Per-episode task labels

The provisional local inventory contains 125 episode JSON files, all with the same `text.goal`: “Pick up the black box on the left at the front using both hands and bring it to the centered carry position.” Visual review found a clear mismatch in episode 0141: both hands lift the box starting on the right at the front. The five completed pilot episodes—0037, 0040, 0042, 0046, and 0047—are consistent with their original left/front instruction. Episode 0141 was not in that pilot.

The user chose to retain accurate per-episode spatial descriptions. The reference for left/right still awaits a choice: the target's position within the visible tabletop/workspace, or its position relative to the other candidate box. Both use the recorded camera view; neither establishes calibrated world coordinates or camera mirroring. These interpretations can disagree. In episode 0081, the wide target occupies the front-left workspace but its center is to the right of the narrow rear box. Front/back must follow the visible table layout, not box size alone, and left/right must not be assigned solely by splitting the image at its center.

The [versioned spatial review](../annotations/spatial-goal-review-v1.json) preserves 125 episode assessments and 337 original-image evidence references, with both direction conventions and source hashes. It is a review artifact, not an accepted override manifest. Final full-dataset label totals and acceptance remain pending the convention choice. The separate [episode 0141 override](../annotations/goal-overrides-0141-v1.json) is an accepted **partial, one-episode overlay**: its right/front correction is valid under both conventions. It does not accept labels for any other episode.

An optional `--label-overrides <reviewed-manifest.json>` argument to [prepare_pilot.py](../scripts/prepare_pilot.py) applies reviewed labels while creating a new snapshot. Without it, snapshot preparation preserves the original goals. The versioned JSON manifest uses `schema_version: 1`, the source directory's `dataset_name`, a stated `reference_frame`, and an `episodes` list. Each episode entry records:

- `episode`: its source folder name, such as `episode_0141`.
- `source_data_json_sha256`: the SHA-256 of the exact original `data.json` bytes.
- `original_goal`: the original instruction, checked against that source file.
- `status`: `accepted` or `needs_review`.
- `goal`: the reviewed instruction, required for an accepted entry; it may equal the original when no correction is needed.

Keep each reviewed manifest revision as a separate artifact. Spatial corrections preserve the rest of the instruction, including the two-handed pickup and centered carry position. Source hashes bind decisions to the recordings that were reviewed; every accepted entry is checked before selection, including entries beyond the requested count. A changed source hash or original goal causes preparation to fail. With an override manifest supplied, only episodes with accepted entries can be selected. Episodes absent from the manifest and entries marked `needs_review` are excluded, alongside existing recording checks and exclusions. Do not turn an unknown position into an accepted label to fill the requested snapshot count.

Episode 0092 remains unresolved because its first available data row points to `colors/000295_color_0.jpg`, with the hands already contacting the box; the precontact setup is missing. Episode 0136 remains excluded pending review of its frozen-image segment, even where its spatial position is clear. Episode 0165 references images that are absent locally, so it has no verified visual label. The completeness of the source copy is still unconfirmed; the 125-file inventory is not a final completeness claim.

Use a fresh `--run` name for a snapshot with reviewed labels. Preparation leaves the original recordings and completed datasets unchanged. Within each copied episode, only the `text.goal` value may change; images, states, actions, timestamps, and other JSON fields are preserved. The new snapshot retains the exact override manifest as `label-overrides.json`, original episode JSON bytes under `provenance/`, and hashes plus original/effective goals in its `manifest.json`. Snapshot verification checks those records and confirms that no episode fields other than `text.goal` changed.

Changed labels require a new conversion from that snapshot using the existing conversion workflow and a fresh dataset identity. Convert to the required LeRobot formats and validate the resulting task text against the new snapshot before using it. Editing an override manifest does not update an already converted dataset or a completed training run. This workflow changes task metadata only; it adds no model or training-pipeline customization.

The episode 0141 overlay was prepared and verified in `derived/label-check-0141-v1`, converted through LeRobot v3 to v2.1 as `local/g1_goal_label_check_0141_v1`, and validated with `--apply-metadata`. All 268 rows retained exact float32 states/actions and the corrected task text; all 268 video frames decoded, with sampled source-image comparisons recorded in `derived/label-check-0141-validation.json`. The 75-test suite passed. Only episode 0141's goal in the new derived copy was corrected; its raw source and the old pilot remain unchanged. No statistics computation, model execution, or training was run for this check.

To reproduce that one-episode check with the default configured paths, use fresh output names as below. Choose another suffix if these names already exist. The final command applies the existing GR00T metadata fields and validates the conversion without computing statistics.

```bash
python3 scripts/prepare_pilot.py \
  --count 1 --run label-check-0141-replay-v1 --max-mib 512 \
  --label-overrides annotations/goal-overrides-0141-v1.json
python3 scripts/prepare_pilot.py \
  --verify derived/label-check-0141-replay-v1
python3 scripts/convert_single_camera.py \
  --pilot-root derived/label-check-0141-replay-v1 \
  --repo-id local/g1_goal_label_check_0141_replay_v1
python3 scripts/convert_single_camera.py --to-v21 \
  --pilot-root derived/label-check-0141-replay-v1 \
  --repo-id local/g1_goal_label_check_0141_replay_v1
python3 scripts/prepare_gr00t_dataset.py \
  --dataset derived/lerobot/local/g1_goal_label_check_0141_replay_v1 \
  --pilot-root derived/label-check-0141-replay-v1 \
  --report derived/label-check-0141-replay-v1-validation.json \
  --apply-metadata
```
