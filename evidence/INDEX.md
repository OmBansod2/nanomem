# Evidence

The 80 files behind the claims this package's documentation makes, so every claim
can be checked rather than taken on trust. Results are JSON as the measuring script wrote
them; the scripts that produced them are here too, so a number can be re-measured
rather than merely re-read.

**These are working records, not a report.** They were written to settle questions
during development and they are kept in that form deliberately — including the
negative results, the measured dead ends and the pre-registered bars that were then
missed. `BENCHMARKS.md` is the curated view; this is what it is built from.

## How to read one

Results files added from 0.7.15 on carry a `measured_on` block naming the nanomem
and engine version they were produced under. **3 of the 59 JSON files here do**;
under `benchmarks/`, 13 of 13 do. `release_preflight.py` refuses to publish a
release whose engine differs from the one a STAMPED file was measured on, and
refuses any unstamped file under `benchmarks/`. It does not yet require a stamp on
the older files here, so an unstamped file in this directory records the build it
was measured on only in its own `date` field.

This paragraph claimed every file carried a stamp through 0.7.20. It did not, and
counting was the whole point of the directory.

Local paths have been replaced with `<repo>/`. No number was altered: the
publishing step parses each file, rewrites only string values and keys, and
asserts the full list of numbers is unchanged before writing.

## What is deliberately not here

| not published | why |
|---|---|
| `clean_chat_benchmark_heldout.json`, `clean_chat_benchmark_persona4.json`, `clean_chat_embeds_heldout.npz`, and the two results files derived from them | held out of development so that scores against them mean something. Publishing them would make every future measurement on them worthless. |
| `assets/` | a 139 MB archived model file. The citation is a note recording where a removed file was kept locally, not evidence for a claim. |
| the `ranking_dev_*.json` family | the round-4 and round-5 dev-persona probes. They carry the recorded query text of those personas, which includes fabricated but realistic contact details, and they are an evaluation corpus whose value depends on not being public. Cited by name in the source so the arms can be identified; not published. |
| `golden/*.dat` | the two expected-result JSONs are published; the vault fixtures are not. `book_v2.dat` is 6.3 MB of ingested book text whose provenance could not be established. |

## Files

### Ranking and revisions

| file | size |
|---|---:|
| `adjacent_attributes_v3r3.json` | 10 KiB |
| `exactness_v3r2.json` | 1 KiB |
| `floor_current_value_results.json` | 7 KiB |
| `floor_retune_results.json` | 4 KiB |
| `grouping_signal_results.json` | 2 KiB |
| `intent_margin_arms_results.json` | 1 KiB |
| `marker_separation_v3r4.json` | 0 KiB |
| `pca_screen_results.json` | 53 KiB |
| `screen_exactness_results.json` | 2 KiB |
| `selection_results.json` | 75 KiB |
| `window_topk_results.json` | 2 KiB |

### Chat and temporal behaviour

| file | size |
|---|---:|
| `chat_target_decision.json` | 79 KiB |
| `clean_chat_results_current_engine.json` | 1 KiB |
| `clean_chat_results_r5_3p.json` | 1 KiB |
| `staleness_calibration.json` | 4 KiB |
| `temporal_as_of_results.json` | 3 KiB |
| `temporal_bench_results.json` | 145 KiB |
| `temporal_cost_results.json` | 1 KiB |
| `temporal_g1_results.json` | 1 KiB |
| `usecases_findings.json` | 4 KiB |
| `write_classifier_v2_results.json` | 41 KiB |
| `write_policy_results.json` | 40 KiB |
| `write_time_anchor_results.json` | 2 KiB |

### Performance, memory and storage

| file | size |
|---|---:|
| `bench_ingest_ram.py` | 58 KiB |
| `bench_memory.py` | 77 KiB |
| `bench_reopen.py` | 76 KiB |
| `bench_screen.py` | 28 KiB |
| `bench_sidecar_size.py` | 37 KiB |
| `bench_turn_latency.py` | 104 KiB |
| `crypto_overhead_v3r3.json` | 1 KiB |
| `ingest_ram_results.json` | 126 KiB |
| `memory_results.json` | 189 KiB |
| `reopen_results.json` | 111 KiB |
| `rewrite_cost_v3r4.json` | 0 KiB |
| `router_gate_results.json` | 118 KiB |
| `router_gate_v3r3.json` | 3 KiB |
| `router_persist_v3r4.json` | 1 KiB |
| `rss_v3r4.json` | 1 KiB |
| `scale_results_current_engine.json` | 1 KiB |
| `scale_results_v3r3.json` | 2 KiB |
| `scale_results_v3r4.json` | 2 KiB |
| `sidecar_size_results.json` | 515 KiB |
| `stats_rss_gap_v3r5.json` | 1 KiB |
| `sweep_router_gate.py` | 60 KiB |
| `sweep_routing_1190.txt` | 3 KiB |
| `turn_latency_results.json` | 602 KiB |

### Comparisons and scorecards

| file | size |
|---|---:|
| `competitors_standard_results.json` | 80 KiB |
| `context_lever_results.json` | 234 KiB |
| `exotic_ranking_results.json` | 204 KiB |
| `exotic_routing_results.json` | 457 KiB |
| `experiment_4d_bridge_results.json` | 22 KiB |
| `final_scorecard.json` | 152 KiB |
| `headtohead_v3.json` | 8 KiB |
| `hybrid_results.json` | 64 KiB |
| `multihop_texthop_v3r2_1190.json` | 0 KiB |
| `quality_summary.json` | 19 KiB |
| `third_party_v3r4.json` | 1 KiB |

### Method, findings and specs

| file | size |
|---|---:|
| `design/DECISIONS.md` | 7 KiB |
| `design/core_spec.md` | 86 KiB |
| `design/edge_cases_spec.md` | 3 KiB |
| `design/floor_current_value_spec.md` | 9 KiB |
| `design/intent_margin_spec.md` | 6 KiB |
| `design/new_usecases_spec.md` | 5 KiB |
| `design/screen_exactness_spec.md` | 4 KiB |
| `design/temporal_api_spec.md` | 4 KiB |
| `finding_floor_drops_current_value.json` | 4 KiB |
| `finding_stale_distribution_copies.json` | 2 KiB |
| `golden/book_v2_golden.json` | 17 KiB |
| `golden/chat_v2_golden.json` | 9 KiB |
| `verify_round3_v3r3.json` | 6 KiB |

### Other

| file | size |
|---|---:|
| `exp_fuzz_ops.py` | 14 KiB |
| `ranking_r5_temporal_layer.json` | 12 KiB |
| `refresh_benchmarks.py` | 13 KiB |
| `train_write_classifier.py` | 24 KiB |

_80 files, 4.2 MiB._
