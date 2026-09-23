#!/usr/bin/env python3
"""Offline bootstrap tests with the fake API executable and real local Git.

Build: zig build-exe push_integration_test.zig -femit-bin=/tmp/push-fixture
Run: python3 integration_tests_push.py /tmp/push-fixture
The Git shim changes only transport, routing pushes to a local bare repository.
"""
import json
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

BINARY = str(Path(sys.argv.pop(1)).resolve())
GIT = shutil.which("git")
URL = "https://api.timbal.ai/orgs/7/projects/99/git"


class PushTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="timbal-push-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "my-project"
        self.repo.mkdir()
        self.home = self.root / "home"
        (self.home / ".timbal").mkdir(parents=True)
        (self.home / ".timbal/config").write_text(
            "[default]\norg=7\n[profile dev]\norg=8\nbase_url=https://api.dev.timbal.ai\n"
        )
        (self.home / ".timbal/credentials").write_text(
            "[default]\napi_key=fake-default\n[profile dev]\napi_key=fake-dev\n"
        )
        self.bare = self.root / "remote.git"
        self.events = self.root / "events"
        self.calls = self.root / "git-calls"
        self.env = dict(os.environ, HOME=str(self.home), USERPROFILE=str(self.home),
                        GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                        PUSH_TEST_EVENTS=str(self.events), PUSH_TEST_CALLS=str(self.calls),
                        PUSH_TEST_BARE=str(self.bare), PUSH_TEST_REAL_GIT=GIT)
        for name in list(self.env):
            if name.startswith("GIT_") and name not in ("GIT_CONFIG_NOSYSTEM", "GIT_CONFIG_GLOBAL"):
                del self.env[name]
        self.env.pop("TIMBAL_PROFILE", None)
        self.git("init", "--initial-branch=feature/import")
        self.git("config", "user.name", "Test")
        self.git("config", "user.email", "test@example.com")
        self.git("commit", "--allow-empty", "-m", "Existing history")
        self.git("init", "--bare", str(self.bare))
        shim = self.root / "bin"
        shim.mkdir()
        (shim / "git").write_text("""#!/usr/bin/env python3
import json, os, subprocess, sys
args = sys.argv[1:]
with open(os.environ['PUSH_TEST_CALLS'], 'a') as f:
    f.write(json.dumps({'args': args, 'profile': os.environ.get('TIMBAL_PROFILE')}) + '\\n')
if args[0] in ('fetch', 'ls-remote'):
    args = [os.environ['PUSH_TEST_BARE'] if x.startswith('https://api.') else x for x in args]
if args[0] == 'push':
    if 'PUSH_TEST_EXIT' in os.environ:
        sys.exit(int(os.environ['PUSH_TEST_EXIT']))
    remote = args[args.index('--') + 1]
    url = subprocess.check_output([os.environ['PUSH_TEST_REAL_GIT'], 'remote', 'get-url', '--push', remote], text=True).strip()
    args = ['-c', 'url.' + os.environ['PUSH_TEST_BARE'] + '.insteadOf=' + url] + args
if args[:3] == ['remote', 'add', 'timbal'] and 'PUSH_TEST_LINK_FAIL' in os.environ:
    sys.exit(1)
sys.exit(subprocess.call([os.environ['PUSH_TEST_REAL_GIT']] + args))
""")
        (shim / "git").chmod(0o755)
        self.env["PATH"] = str(shim) + os.pathsep + self.env["PATH"]

    def git(self, *args, cwd=None, check=True):
        return subprocess.run([GIT, *args], cwd=cwd or self.repo, env=self.env,
                              capture_output=True, text=True, check=check).stdout.strip()

    def push(self, *args, code=0, cwd=None):
        result = subprocess.run([BINARY, *args], cwd=cwd or self.repo, env=self.env,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, code, result.stdout + result.stderr)
        return result

    def creations(self):
        return [json.loads(line) for line in self.events.read_text().splitlines()] if self.events.exists() else []

    def scaffold_main(self):
        scaffold = self.root / "scaffold"
        scaffold.mkdir()
        self.git("init", "--initial-branch=main", cwd=scaffold)
        self.git("config", "user.name", "Timbal", cwd=scaffold)
        self.git("config", "user.email", "noreply@timbal.ai", cwd=scaffold)
        (scaffold / "README.md").write_text("Generated scaffold\n")
        self.git("add", ".", cwd=scaffold)
        self.git("commit", "-m", "Initial scaffold", cwd=scaffold)
        self.git("push", str(self.bare), "main", cwd=scaffold)
        return scaffold, self.git("rev-parse", "HEAD", cwd=scaffold)

    def test_main_bootstrap_replaces_only_scaffold_and_keeps_local_history(self):
        self.git("branch", "-M", "main")
        _, sha = self.scaffold_main()
        tip = self.git("rev-parse", "HEAD")
        self.push()
        self.assertEqual(self.git("--git-dir", str(self.bare), "rev-parse", "main"), tip)
        self.assertEqual(self.git("rev-parse", "HEAD"), tip)
        self.assertEqual(self.git("config", "timbal.bootstrap-url", check=False), "")
        self.assertEqual(self.git("config", "timbal.bootstrap-sha", check=False), "")
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        self.assertIn("--force-with-lease=refs/heads/main:" + sha, [c['args'] for c in calls if c['args'][0] == 'push'][0])
        self.git("commit", "--allow-empty", "-m", "Follow-up")
        self.push()
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        self.assertFalse(any(a.startswith('--force') for a in [c['args'] for c in calls if c['args'][0] == 'push'][-1]))
        self.assertEqual(len(self.creations()), 1)

    def test_main_retry_retains_original_lease_and_rejects_concurrent_update(self):
        self.git("branch", "-M", "main")
        scaffold, sha = self.scaffold_main()
        self.env['PUSH_TEST_EXIT'] = '23'
        self.push(code=23)
        self.assertEqual(self.git("config", "timbal.bootstrap-sha"), sha)
        self.git("commit", "--allow-empty", "-m", "Concurrent change", cwd=scaffold)
        self.git("push", str(self.bare), "main", cwd=scaffold)
        concurrent = self.git("rev-parse", "HEAD", cwd=scaffold)
        del self.env['PUSH_TEST_EXIT']
        self.push(code=1)
        self.assertEqual(self.git("config", "timbal.bootstrap-sha"), sha)
        self.assertEqual(self.git("--git-dir", str(self.bare), "rev-parse", "main"), concurrent)
        self.assertEqual(len(self.creations()), 1)

    def test_main_rejects_modified_or_non_scaffold_history_before_forcing(self):
        self.git("branch", "-M", "main")
        scaffold, _ = self.scaffold_main()
        self.git("commit", "--allow-empty", "-m", "User change", cwd=scaffold)
        self.git("push", str(self.bare), "main", cwd=scaffold)
        self.assertIn("no longer the generated scaffold", self.push(code=1).stderr)
        self.assertEqual(self.git("config", "timbal.bootstrap-sha", check=False), "")

    def test_existing_project_never_gets_automatic_bootstrap(self):
        self.git("branch", "-M", "main")
        _, sha = self.scaffold_main()
        self.git("remote", "add", "timbal", URL)
        self.push(code=1)
        self.assertEqual(self.git("--git-dir", str(self.bare), "rev-parse", "main"), sha)
        self.assertEqual(self.creations(), [])

    def test_bootstrap_dry_run_preserves_pending_lease(self):
        self.git("branch", "-M", "main")
        self.scaffold_main()
        self.env['PUSH_TEST_EXIT'] = '23'
        self.push(code=23)
        before = (self.repo / '.git/config').read_bytes()
        self.push('--dry-run')
        self.assertEqual((self.repo / '.git/config').read_bytes(), before)
        self.push('--force', code=1)

    def test_first_push_links_and_preserves_origin_upstream_and_dirty_files(self):
        self.git("remote", "add", "origin", "git@github.com:example/project.git")
        self.git("config", "branch.feature/import.remote", "origin")
        self.git("config", "branch.feature/import.merge", "refs/heads/feature/import")
        (self.repo / "untracked.txt").write_text("local only")
        self.push()
        self.assertEqual(self.git("remote", "get-url", "timbal"), URL)
        self.assertEqual(self.git("remote", "get-url", "origin"), "git@github.com:example/project.git")
        self.assertEqual(self.git("config", "branch.feature/import.remote"), "origin")
        self.assertEqual(self.git("--git-dir", str(self.bare), "rev-parse", "refs/heads/feature/import"), self.git("rev-parse", "HEAD"))
        self.assertIn("?? untracked.txt", self.git("status", "--porcelain"))
        self.assertEqual(self.creations()[0]["branch"], "feature/import")
        self.push()
        self.assertEqual(len(self.creations()), 1)

    def test_existing_remote_needs_no_create_credentials(self):
        self.git("remote", "add", "origin", URL)
        shutil.rmtree(self.home / ".timbal")
        self.push()
        self.assertEqual(self.creations(), [])
        self.assertEqual(self.git("remote"), "origin")

    def test_profile_and_metadata(self):
        (self.repo / "ui").mkdir()
        (self.repo / "ui/package.json").write_text("{}")
        (self.repo / "workforce/helper").mkdir(parents=True)
        (self.repo / "workforce/helper/timbal.yaml").write_text('_type: agent\n_id: "test-id"\n')
        self.git("add", "ui", "workforce")
        self.git("commit", "-m", "UI")
        self.push("--profile", "dev", "--name", 'A "project"', "--org", "9")
        request = self.creations()[0]
        self.assertEqual(request, dict(org="9", name='A "project"', branch="feature/import",
                                      base_url="https://api.dev.timbal.ai", api_key="fake-dev"))
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        self.assertEqual([c for c in calls if c["args"][0] == "push"][0]["profile"], "dev")
        self.assertNotIn("fake-dev", (self.repo / ".git/config").read_text())

    def test_profile_from_environment(self):
        self.env["TIMBAL_PROFILE"] = "dev"
        self.push()
        self.assertEqual(self.creations()[0]["org"], "8")

    def test_dry_run_does_not_create_or_link(self):
        before = (self.repo / ".git/config").read_bytes()
        (self.home / ".timbal/credentials").unlink()
        self.push("--dry-run")
        self.assertEqual(self.creations(), [])
        self.assertEqual((self.repo / ".git/config").read_bytes(), before)
        self.assertFalse((self.repo / ".git/timbal-push.lock").exists())

    def test_existing_dry_run_does_not_upload(self):
        self.git("remote", "add", "timbal", URL)
        self.push("--dry-run", "--set-upstream")
        self.assertEqual(self.git("--git-dir", str(self.bare), "show-ref", check=False), "")
        self.assertEqual(self.git("config", "branch.feature/import.remote", check=False), "")

    def test_detached_and_unborn_fail_before_create(self):
        self.git("checkout", "--detach")
        self.assertIn("detached", self.push(code=1).stderr)
        self.git("checkout", "--orphan", "new")
        self.assertIn("No commits", self.push(code=1).stderr)
        self.assertEqual(self.creations(), [])

    def test_api_failure_leaves_no_remote(self):
        self.env["PUSH_TEST_API_FAIL"] = "1"
        self.push(code=1)
        self.assertEqual(self.git("remote"), "")

    def test_failed_push_retries_existing_project_and_propagates_exit_status(self):
        self.env["PUSH_TEST_EXIT"] = "23"
        self.push(code=23)
        self.assertEqual(self.git("remote", "get-url", "timbal"), URL)
        del self.env["PUSH_TEST_EXIT"]
        self.push()
        self.assertEqual(len(self.creations()), 1)

    def test_failed_link_recovers_pending_url(self):
        self.env["PUSH_TEST_LINK_FAIL"] = "1"
        self.push(code=1)
        self.assertEqual(self.git("config", "timbal.pending-project-url"), URL)
        del self.env["PUSH_TEST_LINK_FAIL"]
        self.push()
        self.assertEqual(len(self.creations()), 1)
        self.assertEqual(self.git("config", "timbal.pending-project-url", check=False), "")

    def test_non_fast_forward_is_not_forced(self):
        self.push()
        self.git("commit", "--allow-empty", "-m", "Local divergent")
        self.git("commit", "--amend", "--allow-empty", "-m", "Rewrite root")
        # Diverge from the already-pushed root, rather than create its descendant.
        self.git("checkout", "--orphan", "replacement")
        self.git("commit", "--allow-empty", "-m", "Unrelated history")
        self.git("branch", "-M", "feature/import")
        self.assertIn("rejected", self.push(code=1).stderr)
        self.assertEqual(len(self.creations()), 1)

    def test_ambiguous_and_conflicting_remotes_fail(self):
        self.git("remote", "add", "one", URL)
        self.git("remote", "add", "two", URL.replace("99", "100"))
        self.push(code=1)
        self.git("remote", "remove", "one")
        self.git("remote", "remove", "two")
        self.git("remote", "add", "timbal", "git@github.com:example/project.git")
        self.push(code=1)
        self.assertEqual(self.creations(), [])

    def test_multiple_push_urls_are_rejected(self):
        self.git("remote", "add", "timbal", URL)
        self.git("config", "--add", "remote.timbal.pushurl", URL)
        self.git("config", "--add", "remote.timbal.pushurl", "https://other.example/repo.git")
        self.push(code=1)
        self.assertEqual(self.creations(), [])

    def test_timbal_url_after_non_timbal_push_url_is_rejected(self):
        self.git("remote", "add", "origin", "https://github.com/example/repo")
        self.git("config", "--add", "remote.origin.pushurl", "https://github.com/example/repo")
        self.git("config", "--add", "remote.origin.pushurl", URL)
        self.push(code=1)
        self.assertEqual(self.creations(), [])

    def test_concurrent_bootstrap_is_rejected_before_create(self):
        with (self.repo / ".git/timbal-push.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertIn("Another timbal push", self.push(code=1).stderr)
            self.assertEqual(self.creations(), [])
        self.push()

    def test_worktree_subdirectory_uses_repository_link(self):
        worktree = self.root / "linked-worktree"
        self.git("worktree", "add", "-b", "another", str(worktree))
        subdir = worktree / "nested"
        subdir.mkdir()
        self.push(cwd=subdir)
        self.assertEqual(self.git("remote", "get-url", "timbal"), URL)
        self.assertEqual(self.creations()[0]["branch"], "another")
        self.push()
        self.assertEqual(len(self.creations()), 1)

    def test_creation_options_on_linked_repo_are_not_silently_ignored(self):
        self.git("remote", "add", "timbal", URL)
        self.push("--org", "123", code=1)
        self.assertEqual(self.creations(), [])

    def test_git_options_and_explicit_upstream(self):
        self.push("--porcelain", "--atomic", "--follow-tags", "--force-with-lease", "-u")
        self.assertEqual(self.git("config", "branch.feature/import.remote"), "timbal")

    def test_invalid_org_and_untrusted_host_fail_before_create(self):
        self.push("--org", "../../bad", code=1)
        (self.home / ".timbal/config").write_text("[default]\norg=7\nbase_url=https://evil.example\n")
        self.push(code=1)
        self.assertEqual(self.creations(), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
