# Releasing

`main` is where everything is developed, and it is heading to **3.0**: nothing is published from `main` until
3.0 ships. Meanwhile the **2.x line is released from release branches**, so fixes, model updates, new tools
and docs reach 2.x users without waiting for 3.0 or carrying it.

## The model

- **Everything lands on `main` first,** through a normal PR. A release branch only receives *backports*:
  a merged PR cherry-picked onto it, through a PR of its own. Nothing is developed on a release branch, and
  nobody pushes to one directly.
- **What goes to 2.x:** what 2.x users can use and that does not depend on 3.0 work — bug fixes, model
  catalog updates, new tools and integrations (Notion, Qlik, ...), docs, and whatever CI needs to stay green
  there. Not 3.0 features or refactors: those stay on `main`.
- **One active 2.x branch at a time.** Today it is `release/2.8`, and every `2.8.x` is tagged on it.
- **Patch or minor, decided per release:** fixes, model updates and docs are the next patch of the active
  branch (`v2.8.6`). For a release that adds enough to deserve a minor, cut `release/2.9` **from
  `release/2.8`, not from `main`** (`main` holds 3.0), and tag `v2.9.0` there. `release/2.9` is then the
  active branch and `release/2.8` is retired.
- **3.0** is tagged on `main` (`v3.0.0`) when it ships; `release/3.0` is cut from that tag only when the first
  `3.0.x` patch is needed. From then on 2.x takes critical fixes only.

```
main (3.0) ──●──●──●──●──●──●──●──●──●──●──●──●──●──   v3.0.0.dev0 tag: main builds 3.0.0.devN
              (every PR merges here first)
release/2.8:  v2.8.3 ── #276 ── #284 ── #288 ── #298 ── #303 ── v2.8.5 ── <backports> ── v2.8.6
                                    (backports: cherry-picks of merged PRs)
release/2.9 (when needed, cut from release/2.8): <backports> ── v2.9.0 ── ...
```

## Versions

Versions come from git tags (hatch-vcs), with `version_scheme = "release-branch-semver"` in `pyproject.toml`:

| Where | Builds as |
| --- | --- |
| `main`, feature branches | `3.0.0.devN+g<hash>` (from the `v3.0.0.dev0` tag on `main`) |
| `release/2.8` | `2.8.6.devN+g<hash>`: the next **patch** of that line |
| `release/2.9`, cut from `release/2.8` | `2.9.0.devN+g<hash>` until `v2.9.0` is tagged on it |
| a tag `v2.8.6` | exactly `2.8.6` |

- A release branch has to be named `release/<major>.<minor>` for this to work.
- Put a version tag on a commit that has no other version tag: when two are on one commit, the build picks
  either. So tag `v2.9.0` after the first backport into `release/2.9`, not on the commit it was cut at.
- **PyPI never accepts a version twice, even after it is deleted.** If a version is deleted from PyPI, delete
  its tag as well and release the next number (2.8.4 was deleted; the line went on with 2.8.5).

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
release line what the PR does on `main`, and say in the commit what you changed and why. Parts of the PR that
change code the line does not have (3.0 subsystems) are left out. When a PR needs another one that is not on
the line, backport that one first, or leave it out.

Review and merge the backport PR as usual (release branches have no merge queue: the normal Merge button).

## Cutting a release

From the active release branch, once CI is green on it:

```bash
git fetch origin
git tag v2.8.6 origin/release/2.8
git push origin v2.8.6
scripts/release.sh 2.8.6
```

`release.sh` writes release notes from the commits since the previous tag on that branch, builds the CLI
binaries (zig), creates the GitHub release (the installer's "latest"), then asks before publishing the Python
package to PyPI (`scripts/publish-pypi.sh`, which builds from a clean checkout of the tag). Publishing needs
`UV_PUBLISH_TOKEN` (or `uv publish`'s prompts); `scripts/publish-pypi.sh <VERSION> --dry-run` checks
everything without uploading. Run it from a checkout of the release branch, so its own scripts are used.

On macOS, zig 0.14 cannot link against the macOS 26 SDK (Xcode 26): `scripts/zig_macos_sdk.sh` builds
against the Command Line Tools' SDK instead, and says so. If neither SDK works it stops with an explanation.

## CI

`ci.yml` and `perf.yml` run on PRs into `main` and into `release/**`. The perf gate compares a backport
against the release branch it targets. To run CI on a branch without a PR:
`gh workflow run ci.yml --ref release/2.8`.

GitHub settings (rulesets):

- **`main`:** a merge queue. A pull request runs the fast lane (one job); the queue runs every OS on what is
  about to land before merging. A PR, or a queue group, that changes nothing the tests read (docs, release
  scripts, other workflows) runs no tests.
- **`release`** (`release/**`): GitHub's merge queue takes no branch pattern, so a backport PR runs every OS
  on each push instead (unless it changes nothing the tests read), and needs one approval and `CI result`.
- Both: no direct pushes, force pushes or deletions, and no bypass.
