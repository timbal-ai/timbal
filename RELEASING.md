# Releasing

`main` is where everything is developed. Releases come either from `main` (a new minor version) or from a
release branch (patch versions of a line that already shipped), so a release can carry some of what is on
`main` without waiting for all of it.

## The model

- **One release branch per minor line:** `release/2.8`, `release/2.9`, ... Every `2.8.x` is tagged on
  `release/2.8`.
- **Everything lands on `main` first,** through a normal PR. A release branch only receives *backports*:
  a merged PR cherry-picked onto it, through a PR of its own. Nothing is developed on a release branch.
- **A new minor version is tagged on `main`** (`v2.9.0`). Its release branch is cut from that tag only when
  the first `2.9.x` patch is needed. Older lines then take critical fixes only, and are retired.

```
main ──●──●──●──●──●──●──●──●──●── v2.9.0 ──●──●──
        \     (PRs merge here first)        \
         release/2.8: v2.8.3 ── #288 ── #276 ── #284 ── v2.8.4 ── #300 ── v2.8.5
                                 (backports: cherry-picks of merged PRs)
```

## Versions

Versions come from git tags (hatch-vcs), with `version_scheme = "release-branch-semver"` in `pyproject.toml`:

| Where | Builds as |
| --- | --- |
| `main`, feature branches | `2.9.0.devN+g<hash>`: the next **minor**, never a number a release line will publish |
| `release/2.8` | `2.8.5.devN+g<hash>`: the next **patch** of that line |
| a tag `v2.8.5` | exactly `2.8.5` |

A release branch has to be named `release/<major>.<minor>` for this to work.

## Backporting a PR

Once the PR is merged into `main`:

```bash
scripts/backport.sh <pr-number> [<line>]      # e.g. scripts/backport.sh 300 2.8 (default: the newest release/*)
```

It cherry-picks the PR's merge commit onto `backport/<pr>-to-<line>` (one commit, titled like the PR and
ending in `(cherry picked from commit <sha>)`), pushes it and opens a PR into `release/<line>`, where CI runs.
It refuses a PR that is not merged into `main` or is already on the branch. `BACKPORT_DRY_RUN=1` does
everything but the push and the PR.

On a conflict it stops and prints the commands to finish by hand. Resolve it so the backport does on the
release line what the PR does on `main`, and say in the commit what you changed and why. When a PR needs
another one that is not on the line, backport that one first, or leave it out.

Review and merge the backport PR as usual. **Never push to a release branch directly.**

## Cutting a release

From an up-to-date release branch (patch) or `main` (minor), once CI is green on it:

```bash
git tag v2.8.5 origin/release/2.8        # or: git tag v2.9.0 origin/main
git push origin v2.8.5
scripts/release.sh 2.8.5
```

`release.sh` writes release notes from the commits since the previous tag on that branch, builds the CLI
binaries (zig), creates the GitHub release, then asks before publishing the Python package to PyPI
(`scripts/publish-pypi.sh`, which builds from a clean checkout of the tag). Publishing needs
`UV_PUBLISH_TOKEN` (or `uv publish`'s prompts); `scripts/publish-pypi.sh <VERSION> --dry-run` checks
everything without uploading. **PyPI never accepts a version twice, even after it is deleted.**

## CI

`ci.yml` and `perf.yml` run on PRs into `main` and into `release/**`. The perf gate compares a backport
against the release branch it targets. To run CI on a branch without a PR:
`gh workflow run ci.yml --ref release/2.8`.

Recommended GitHub settings: protect `release/**` like `main` (a PR and passing CI to merge, no direct pushes).
