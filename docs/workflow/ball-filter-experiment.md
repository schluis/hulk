# reacquisition experiment

Parent: `dev/ball-filter-lab`.

Source snapshot: `archive/pre-lab-output-reacquisition-20261004`.

This port retains the lab baseline safeguards and tooling. Historical archive scores do not validate this combination; evaluate on common recordings with the same scoring and settings before promotion.

Variant parameters: `output_reacquisition_distance`, `output_reacquisition_blend`.

The archived output-reacquisition guard was disabled by default, and remains so:
`output_reacquisition_distance` and `output_reacquisition_blend` are zero. Enable
both explicitly in a temporary parameter layer for an experiment. This port does
not promote a previously rejected parameter configuration to the normal filter.
