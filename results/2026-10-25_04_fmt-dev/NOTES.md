# 04 · Regex-constrained function names (2026-10-25) — negative result

**Question:** experiment 03 showed Qwen writes 43% of its function names in the wrong format (`pkg.mod.Class.func` instead of `pkg/mod.py::Class.func`). Does forcing the format with the decoding grammar fix this?

**Setup:** identical to 03 (dev split, 23 tasks, file tools, 20 steps) plus `--func-format`: in `submit`, each function entry must match the regex `^[A-Za-z0-9_/.-]{1,100}[.]py::[A-Za-z_][A-Za-z0-9_.]{0,60}$`, and file entries must be plain repo paths. Only the final `submit` step differs from 03; all exploration steps are identical (temperature 0).

| Function@5 (scored with name resolution) | 03 (free format) | 04 (regex format) |
|---|---|---|
| Gemma 4 E2B | 43% | 43% |
| Qwen3.5-4B | **62%** | **33%** |
| Qwen malformed names | 43% | 0% |

**Finding: the format constraint destroyed correct answers.** Qwen begins its preferred dotted form (`pvlib.iam.physical`); the grammar does not let the string end without `.py::`, so the model fills it with junk until the length bound forces the suffix:

```
03: pvlib.iam.physical
04: pvlib.iam.physical.physical_1_interface_2_interface_3_interface_..._.py::physical
03: pydicom.dataset.Dataset.to_json_dict
04: pydicom.dataset.Dataset.to_json_dict__method__to_json_dict__method__..._.py::to_json_dict
```
All 6 tasks Qwen lost (03 → 04) fail this way. Gemma already used the requested format, so the constraint changed nothing for it.

The well-formedness metric went to 0% malformed while accuracy halved: **a grammar can guarantee the form of an answer while corrupting its content**, when the forced form diverges from the form the model is mid-way through writing. An unbounded pattern was worse still: in a pre-run test, generation ran on until the token limit.

**Consequence:** the ladder continues with a *structured* format instead (`{"file": ..., "name": ...}` objects, experiment 05), which gives the model a separate slot for each part rather than one regex over a combined string. A partial run of the old ladder on top of 04 (1 task) was discarded.

**Files:** `<model>.json`, `metrics.json`, `config.json`, `figures/summary.*`, `figures/outcomes.*`. Trajectories: `C:\gemma_work\runs\2026-10-25_04_fmt-dev\`.
**Regenerate:** `python scripts/rescore.py && python analysis/plot_experiment.py --exp fmt-dev`.
