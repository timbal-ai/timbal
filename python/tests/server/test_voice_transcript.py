"""Execute the playground's real transcript handlers with a minimal DOM in Node."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is required for playground JavaScript tests")
def test_assistant_acting_tags_across_deltas_and_interruptions():
    html = (Path(__file__).parents[2] / "timbal/server/voice.html").read_text(encoding="utf-8")
    helpers = html[html.index("const VOICE_ACTING_TAGS") : html.index("let partialEl = null;")]
    handler = html[html.index("function handleServerMessage(msg)") : html.index("const LIVEKIT_CLIENT_ESM")]
    harness = r"""
const assert = require('node:assert/strict');
const window = {_sawUserCommit: true};
const document = {createElement: () => ({textContent: '', remove() {this.removed = true;}})};
const transcriptEl = {children: [], appendChild(el) {this.children.push(el);}, scrollHeight: 0};
let suppressAssistantAudio = false;
function updateEmptyState() {}
function setPartial() {}
function setStatus() {}
function stopThinkingSound() {}
function stopAssistantAudio() {}
"""
    checks = r"""
for (const tag of ['relieved', 'laughs', 'curious', 'sighs']) {
  const text = `[${tag}] Good, that's what matters.`;
  for (let split = 1; split < text.length; split++) {
    window._assistantEl = null;
    window._replyDone = true;
    handleServerMessage({type: 'agent_text_delta', text: text.slice(0, split)});
    if (split <= tag.length + 2) assert.equal(window._assistantEl.textContent, '');
    handleServerMessage({type: 'agent_text_delta', text: text.slice(split)});
    assert.equal(window._assistantEl.textContent, "Good, that's what matters.");
    handleServerMessage({type: 'agent_text_done', text});
    assert.equal(window._lastAssistantEl.textContent, "Good, that's what matters.");
  }
}
assert.equal(assistantTranscriptText('Hi [laughs] there [sighs].'), 'Hi there .');
assert.equal(assistantTranscriptText('Order [901] and [customer]'), 'Order [901] and [customer]');
assert.equal(assistantTranscriptText('[warm](https://example.com)'), '[warm](https://example.com)');
assert.equal(log('user', '[laughs] literal user speech').textContent, '[laughs] literal user speech');
assert.equal(log('assistant filler', '[curious] Let me check.').textContent, 'Let me check.');
window._assistantEl = null;
window._replyDone = true;
handleServerMessage({type: 'agent_text_delta', text: '[rel'});
assert.equal(window._assistantEl.textContent, '');
handleServerMessage({type: 'agent_text_done', text: '[rel'});
assert.equal(window._lastAssistantEl.textContent, '[rel'); // incomplete literal is not lost
window._replyDone = true;
handleServerMessage({type: 'agent_text_delta', text: '[relieved] Hello there.'});
handleServerMessage({type: 'interrupted', heard_text: '[relieved] Hello'});
assert.equal(transcriptEl.children.at(-1).textContent, 'Hello');
window._replyDone = true;
handleServerMessage({type: 'agent_text_delta', text: '[relieved] Hello there.'});
handleServerMessage({type: 'interrupted', heard_text: '[relieved]'});
assert.equal(transcriptEl.children.at(-1).removed, true);
window._replyDone = true;
handleServerMessage({type: 'agent_text_delta', text: '[laugh'});
handleServerMessage({type: 'interrupted', heard_text: ''});
handleServerMessage({type: 'agent_text_delta', text: 'A fresh reply.'});
assert.equal(window._assistantEl.textContent, 'A fresh reply.');
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", harness + helpers + handler + checks],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
