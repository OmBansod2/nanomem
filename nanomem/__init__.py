"""
nanomem
~~~~~~~
A single-file persistent memory engine for AI applications: append-only storage,
exact in-RAM cosine search, entity-aware temporal ranking, and optional
password protection. numpy is the only third-party runtime dependency.
"""

from .vault import Vault, TextHopBridge
from .users import list_users, create_user, delete_user, user_exists
from .crypto import THREAT_MODEL
from .engine import ENGINE_VERSION
from .errors import (NanomemError, CorruptContainerError, IntegrityError,
                     WrongPasswordError, PasswordRequiredError, ReadOnlyVaultError,
                     ContainerReplacedError, NotEncryptedError, ClosedVaultError,
                     VaultShrankError)

# 0.3.0, not 0.1.0. The wheel that used to ship alongside this source
# (nanomem-0.1.0-py3-none-any.whl) was the v2 engine -- no crypto.py, no
# container.py, `encrypted_at_rest: True` hard-coded over a keyless XOR -- and it
# carried the SAME __version__ string as this source, so nobody could tell which
# one they had. Any 0.1.0 install is the v2 engine; 0.3.0 and above is this one.
# That wheel has since been deleted everywhere it existed and rebuilt as
# nanomem-0.3.0-py3-none-any.whl from this source. Nothing writes the version
# twice any more: pyproject.toml and setup.py both read the string below.
#
# 0.3.1 (engine 3.0.4) for the same reason the string exists at all: the DEFAULT
# behaviour changed. "What was my original address?" returns a different record
# than it did under 0.3.0 with no argument change, and the arena is pre-sized on
# open. A version that did not move would leave two engines answering the same
# call differently under one name. (The stale 0.3.0 wheel this note used to
# warn about was rebuilt and removed at 0.6.0; see the 0.6.0 entry.)
#
# 0.3.2 (engine 3.0.5) for the same reason again, one level down: the arena's
# ALLOCATION policy changed. No answer moves -- the full fp32 score vectors are
# bitwise identical and not one of 500 top-10 lists changes at either corpus
# size -- but a 71,433-document load now peaks at 286.4-286.9 MB of ru_maxrss
# where 0.3.1 peaked at 818.6-819.1 MB, `stats()["arena_bytes"]` stopped
# overstating the arena by up to 1.83x, and `reserve_additional_rows()`,
# `--expect-docs` and three `stats()` keys are new. See CHANGELOG.md 0.3.2 and
# evidence/{ingest_ram_results,memory_results}.json.
#
# 0.4.0 (engine 3.1.0) is a MINOR bump, not a patch one, for three reasons that
# a patch number would hide. (1) A second FILE now appears beside the vault by
# default: `<vault>.arena`, 257.5 MiB beside a 148.7 MiB vault at 71,433 rows.
# Nothing about the vault changes -- `format_version` is still 3, a vault
# written by 0.3.x opens unchanged, and `arena_cache="off"` restores 0.3.2's
# behaviour exactly -- but a default that writes a new file is not a patch.
# (2) Published top-10 lists can REORDER against 0.3.2: `_select_top_k` now
# breaks ties on the row id instead of inheriting `np.argpartition`'s
# unspecified order. Every changed position is a bitwise tie (max |score gap|
# 0.0 over 8 of 500 lists at 71,433 rows, mechanically checked in
# evidence/reopen_results.json :: tie_forensics), so both orders were
# always correct -- but a caller diffing top-10 output against 0.3.2 will see
# the change, which is exactly what a version string is for. (3) Two new
# constructor flags, `arena_cache=` and `screen=`.
# (The wheel this note used to warn about was rebuilt at 0.6.0.)
#
# 0.5.0 (engine 3.2.0) is a MINOR bump for two default changes, either of which
# would have needed one on its own.
# (1) THE SIDECAR STOPPED COPYING THE VAULT. `<vault>.arena` now stores OFFSETS
# into the vault's own fp16 vector runs and record sections instead of a second,
# fp32 copy of them: 257.48 -> 4.38 MiB at 71,433 rows, and a cached vault
# 406.15 -> 153.05 MiB, which is below sqlite-vec's 258.0 and takes back the
# disk axis 0.4.0 had given up. The arena CACHE format goes 2 -> 3, so every
# existing `.arena` is refused once and rebuilt (one slow open, once); the VAULT
# format is still 3 and no vault file changes. Scores are bitwise identical in
# 36 of 36 arm x corpus x screen comparisons. It is not free: the upcast that
# used to be a mapped file is now anonymous memory, so a serving process's
# `phys_footprint` reads 270.5 MB where the copying layout read 43.5 MB, and a
# vault truncated out-of-band UNDER a live engine now raises `VaultShrankError`
# where the copying layout could not notice at all. `arena_cache_vectors="cache"`
# with `arena_cache_records="cache"` restores 3.1.0 exactly.
# (2) THE WRITE GATE'S DECISION POINT MOVED, 0.60 -> 0.05. `Vault.should_store`
# and `add_conversation_turn` now keep about 57% more turns with no argument
# change, worth +13.3 pt of end-to-end top-1 [CI +8.7, +18.3] on a 300-question
# held-out split: the old point was chosen on symmetric accuracy/F1, while in
# deployment a dropped fact is unrecoverable and a kept one costs ~1.9 kB. See
# `nanomem.classifier.DEPLOYMENT_THRESHOLD_FULL` and CHANGELOG 0.5.0;
# `WriteClassifier(threshold=0.60)` restores the old gate exactly.
#
# 0.6.0 (engine 3.3.0) is a MINOR bump for three ADDITIONS. Nothing existing
# changes: `search()` with no `as_of` is bitwise identical to 0.5.0 on 520
# recorded query results, and the whole 0.5.0 suite passes untouched
# (evidence/design/temporal_api_spec.md gate G1).
# (1) `VaultEngine.history()` / `Vault.history()` return the revision CHAIN for
# a fact -- every value it has held, oldest first, with the current one last.
# This is the group `_resolve_revisions` has always assembled in order to decide
# which member to surface, returned instead of discarded; it applies no boost,
# and it deliberately does NOT apply the ranker's relevance floor, because that
# floor drops members and a truncated chain reports a superseded value as
# current (see the docstring and finding_floor_drops_current_value.json).
# (2) `search(as_of=<unix ts>)` answers as the vault stood at that moment. It
# forces an exhaustive scan and turns the PCA screen off, because masking a
# routed or screened shortlist would drop the very record the query wants.
# Proven equal to a physically truncated vault: the admitted row set is exactly
# {rows : ts <= t} over 705 checks, and 5,670 comparisons show 0 id differences,
# 0 order differences and 0 leaks (evidence/temporal_as_of_results.json).
# (3) `changes(since, until)` reports what was written in a window with no query
# vector, reading the resident timestamp column.
# Cost of the added branch, measured over 5 alternating paired runs at 20,000
# rows: +0.21% on p50 against a pre-registered bar of +1.0%
# (evidence/temporal_cost_results.json).
# The CLI gains `nanomem history`, `nanomem changes`, and `--as-of` on search.
# (4) `volatility()` reports how often each fact actually changes, measured off
# the revision log; `staleness()` adds a modelled `p_superseded` that is
# SUPPRESSED unless `assume_memoryless=True`, because the model only ties a
# constant-rate baseline off its own assumption (staleness_calibration.json).
# (5) `nanomem.mcp` is now part of THIS package. It existed only inside one
# distribution bundle through 0.5.x, offered add/search/stats alone, and so
# presented nanomem to every MCP client as an ordinary vector store; it now
# exposes history / as_of / changes / volatility as well, answers a JSON-RPC
# notification with silence instead of a null-id reply, and reports the real
# __version__ instead of a hard-coded "0.1.0".
# DISTRIBUTION: at 0.6.0 the five bundled copies of this package -- which had
# been left at 0.3.0 / engine 3.0.3 for four releases -- were resynced to this
# source and all twelve wheels in the tree were rebuilt from it.
__version__ = "0.7.22"
__all__ = [
    "Vault", "TextHopBridge", "list_users", "create_user", "delete_user",
    "user_exists", "NanomemError", "CorruptContainerError", "IntegrityError",
    "WrongPasswordError", "PasswordRequiredError", "ReadOnlyVaultError",
    "ContainerReplacedError", "NotEncryptedError", "ClosedVaultError",
    "VaultShrankError",
    "THREAT_MODEL", "ENGINE_VERSION", "__version__",
]
