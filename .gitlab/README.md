# Internal GitLab CI

Internal GitLab CI is separate from the GitHub Actions levels in `docs/ci-cd.md`.
Merge request pipelines run lint checks and build a private RL-Gym image through
`nemo-ci`; the image is scoped to the MR and uses the target branch and prior MR
image as cache sources. These builds do not run RL tests. Pushes to protected
branches run lint and refresh that branch's RL-Gym image. Pushes to branches
with an open MR use the MR pipeline, avoiding duplicate runs.

## Post-training script suites

Nightly and manually selected post-training suites are configured in
`tests/ci/posttraining_script_suites.yml`. Add each launcher under the desired
suite's `tests` list with a stable job name and repository-relative script path.
The suite defines its Post-Training repository and default ref, cluster, Slurm
account, partition, QoS, and launcher overrides. These suites are selected by
nemo-ci pipeline variables and do not run on merge requests by default.

Use `POSTTRAINING_SCRIPT_SUITE` to select a suite and `POSTTRAINING_REF` to
override its configured Post-Training branch or tag. Changes to the shared
manifest schema or dynamic job generator belong in `nemo-ci`; adding a script
to an existing suite only requires an update to this manifest.
