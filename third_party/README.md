# Third-party sources

The directories below are Git subtrees. Edit their files directly and commit in
this repository. The benchmark commit identifies the complete source version.
Training provenance records that commit for each component and scopes its dirty
status to the corresponding directory.

Each subtree starts with a squashed upstream baseline. LAGEN adaptations are
separate commits above that baseline, so upstream updates can preserve them.
Original licenses and runtime assets remain in each directory.

## Source baselines

These are the original upstream commits and their deterministically filtered
counterparts. Update both values after accepting an upstream merge. The initial
fork revisions identify the LAGEN adaptations imported during migration.

### MIKASA-Robo

- Directory: `third_party/MIKASA-Robo`
- Upstream: https://github.com/CognitiveAISystems/MIKASA-Robo.git
- Branch: `main`
- Upstream baseline: `509b875f3d207c287497c0a897661062de928bb0`
- Filtered baseline: `c68a0fea2d0231118f7f086d5f10c417cf0a8e64`
- Initial LAGEN snapshot: `01424d175893c40578f6e4b8b2e8d7cc3ece2ee8` from https://github.com/mindorigin150/MIKASA-Robo.git

### flappy-bird-gymnasium

- Directory: `third_party/flappy-bird-gymnasium`
- Upstream: https://github.com/markub3327/flappy-bird-gymnasium.git
- Branch: `main`
- Upstream baseline: `e7eb59e2a850d975f841b5b8e7cdcd93a1f6da7e`
- Filtered baseline: `8442aa15502d612e2fd2a966f5c65cea0bb0e21c`
- Initial LAGEN snapshot: `cd1c7d51836d72229a6bfd00120d30540da4db38` from https://github.com/mindorigin150/flappy-bird-gymnasium.git

### sample-factory

- Directory: `third_party/sample-factory`
- Upstream: https://github.com/alex-petrenko/sample-factory.git
- Branch: `master`
- Upstream baseline: `8b354943a8b04e2fbb7c7c8f35ea3fed195dd631`
- Filtered baseline: `7cc55e8eff0afe59241bd994d059b097ad2f04fb`
- Initial LAGEN snapshot: `ad256a6312db1ff46b19f539081abdbdf20fc424` from https://github.com/mindorigin150/sample-factory.git

### starVLA

- Directory: `third_party/starVLA`
- Upstream: https://github.com/starVLA/starVLA.git
- Branch: `starVLA`
- Upstream baseline: `5ce9e2ed59dc94de926c335d36a3681c243a58f7`
- Filtered baseline: `11432c056fd96dc92b60622d5420394abc7fe973`
- Initial LAGEN snapshot: `af9ebdf66d89d5bd4a0383058a62e8c092ead73c` from https://github.com/talha1503/starVLA.git

### humanoid-bench

- Directory: `third_party/humanoid-bench`
- Upstream: https://github.com/carlosferrazza/humanoid-bench.git
- Branch: `main`
- Upstream baseline: `cb1189039151c8aadaaa987b442da54383c87fab`
- Filtered baseline: `e31942b2947ac15795f20ba1adea8e68a6981a94`
- Initial LAGEN snapshot: `bfb63711ebe41bc0965287353b657f9bc52699da` from https://github.com/mindorigin150/humanoid-bench.git

## Resource filters

The filter removes presentation media with `.gif`, `.mp4`, `.webm`, `.mov`,
`.avi`, or `.mkv` extensions, case-insensitively. It also removes:

- Flappy: `flappy_bird_gymnasium/assets/model/`, containing upstream DQN example weights.
- MIKASA: `.gitmodules` and the `utils/convert_npz_to_rlds/rlds_dataset_builder` gitlink. These upstream Git links are excluded from imported source trees.

LAGEN removes the Flappy DQN entrypoint and documentation embeds after import.
Flappy sprites/audio and HumanoidBench models, meshes, and textures remain.
Media in the benchmark's own documentation and website are outside this filter.

## Update an upstream dependency

Use Git and `git-filter-repo==2.47.0`. Start from a clean benchmark worktree.
Set the source URL and branch from the entry above.
The following example updates MIKASA. Change those three variables for another
source; the path-specific filter cases below remain the same.

```bash
BENCH_ROOT="$(git rev-parse --show-toplevel)"
SOURCE_PREFIX=third_party/MIKASA-Robo
UPSTREAM_URL=https://github.com/CognitiveAISystems/MIKASA-Robo.git
UPSTREAM_BRANCH=main
SOURCE_TMP="$(mktemp -d)"

git clone --bare --no-tags --single-branch --branch "$UPSTREAM_BRANCH" \
  "$UPSTREAM_URL" "$SOURCE_TMP/upstream.git"
NEXT_SOURCE="$(git -C "$SOURCE_TMP/upstream.git" rev-parse "$UPSTREAM_BRANCH")"

FILTER_ARGS=(
  --force --prune-empty never --prune-degenerate never --preserve-commit-hashes
  --invert-paths --path-regex '(?i).*\.(gif|mp4|webm|mov|avi|mkv)$'
)
case "$SOURCE_PREFIX" in
  third_party/flappy-bird-gymnasium)
    FILTER_ARGS+=(--path flappy_bird_gymnasium/assets/model/)
    ;;
  third_party/MIKASA-Robo)
    FILTER_ARGS+=(--path .gitmodules --path utils/convert_npz_to_rlds/rlds_dataset_builder)
    ;;
esac
git -C "$SOURCE_TMP/upstream.git" filter-repo "${FILTER_ARGS[@]}"
NEXT_FILTERED="$(git -C "$SOURCE_TMP/upstream.git" rev-parse "$UPSTREAM_BRANCH")"

git -C "$BENCH_ROOT" subtree pull --prefix="$SOURCE_PREFIX" \
  "$SOURCE_TMP/upstream.git" "$UPSTREAM_BRANCH" --squash
printf 'Upstream baseline: %s\nFiltered baseline: %s\n' "$NEXT_SOURCE" "$NEXT_FILTERED"
```

Resolve merge conflicts normally and keep the existing LAGEN behavior. Remove
any new embeds or DQN entrypoints that the update introduces. Run the relevant
existing tests, update the two baseline values above, and commit the reviewed
changes. Remove the temporary source clone with `rm -rf "$SOURCE_TMP"` after the
merge and checks finish. Do not import unfiltered source trees: deleting media in
a later commit still leaves those files in Git history.
