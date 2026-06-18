# banditevolve

A frozen-model evolutionary code search. The model is never trained. All
adaptation lives in two stacked bandits:

1. **UCT over parents** picks which archived program to expand next.
2. **Per-band Thompson sampling over mutation prompts** picks how to mutate it.

Both are driven by a per-iteration reflection step that reads the rollouts,
finds the dominant failure mode, and grows the mutation-prompt pool of the
relevant score band.

This is built from scratch but reuses the genuinely reusable infrastructure from
the TTT-local codebase. It deliberately drops everything tied to weight training,
because this algorithm does not train the model.

## What was reused vs dropped from TTT

Reused: `sandbox.py` (subprocess runner with hard timeout, near verbatim), the
code extractor from `reward.py`, the `State` + lineage idea and the UCT/lineage
blocking logic from `sampler.py`, and the problem ABC shape from `problems/`.

Dropped: `advantage.py`, `train_multy.py`, `gen_workers.py`, `model_backend.py`,
the whole `reranker/` package, and all LoRA/Unsloth/PEFT machinery. None of it
applies once the model is frozen. The Elo reranker in particular is replaced in
spirit by the per-band prompt bandit plus reflection: instead of re-ranking
parents with an LLM judge, the LLM judge writes new mutation operators.

## Directory layout

```
banditevolve/
├── README.md
├── main.py              entry point
├── config.py            Config dataclass + loader (defaults < yaml < CLI)
├── configs/
│   └── circle_packing.yaml
├── engine.py            the search loop; the 9 steps live here
├── archive.py           State + Archive: UCT select_parents, lineage, sterile/invalid
├── bands.py             BandAssigner (quantile) + BandStats (within-band z-score)
├── prompt_bandit.py     PromptBandit: NIG Thompson sampling over prompt arms
├── mutation.py          build the mutation chat messages
├── validation.py        the child gate (parse / entrypoint / static / no-op / novelty)
├── codetools.py         AST parse, normalized dump, similarity, entrypoint discovery
├── evaluation.py        multi-seed eval -> per-seed rewards -> paired (dmu, dsigma)
├── reflection.py        end-of-iteration LLM reflection -> new band prompt
├── prompts.py           band freedom, seed arm templates, evolution block, reflection
├── llm.py               frozen-model client (OpenAI-compatible + offline dummy)
├── sandbox.py           reused subprocess sandbox
├── reward.py            reused code extractor
└── problems/
    ├── base.py          Problem ABC: build_prompt, build_seed_prompt, preprocess,
    │                    score, static_check, seed_states, run_one
    ├── registry.py      get_problem; circle_packing implemented, others port over
    └── circle_packing.py worked example (validator byte-for-byte from TTT)
```

## The algorithm, step by step, and where each lives

1. **Generate n valid independent seeds and evaluate them.**
   `engine.Engine._bootstrap_seeds` -> `llm` + `validation.quick_gate` +
   `evaluation.evaluate_candidate`. Seeds enter the archive as roots.

2. **Each iteration, sample n parents from the archive via UCT, no repetition.**
   `archive.Archive.select_parents`. Q is the max child reward (best outcome,
   not average), min-max normalized so the exploration term is comparable across
   problems with different reward scales. Lineage blocking stops one iteration's
   parents from collapsing onto a single thread.

3. **Each parent gets k rollouts.** `engine.Engine._iteration` (the `n*k = 16`
   example is `num_parents=4`, `rollouts_per_parent=4`).

4. **Per rollout: with prob epsilon explore (fresh seed), else mutate.**
   `_do_explore` (generate + gate + evaluate a from-scratch program) vs
   `_do_mutate`.

5. **Four bands, each owning a pool of mutation prompts treated as a bandit;
   Thompson sampling picks a prompt; differential mutation freedom per band.**
   `bands.BandAssigner` assigns the band by quantile; `prompt_bandit.PromptBandit`
   is the per-band NIG Thompson-sampling bandit; `prompts.BAND_FREEDOM` enforces
   the freedom difference structurally so it does not depend on which arm is
   sampled.

6. **Mutation prompt contains parent, grandparent, archive-best, worst-but-valid,
   and the parent's band.** `mutation.build_mutation_messages` +
   `prompts.evolution_block`, appended onto the problem's own task prompt.

7. **Child validation gate.** `validation.validate_child`: parse, required
   entrypoint, problem static constraints, no-op/cosmetic vs parent, novelty vs
   {parent, grandparent, best, top-10}. Failing novelty archives the child as
   STERILE; failing a hard rule archives it as INVALID. Both are non-parents.

8. **Evaluate the child on the SAME seeds as the parent; compute paired
   mean-delta and sigma-delta.** `evaluation.evaluate_candidate` over
   `parent.per_seed` keys, then `evaluation.paired_delta` -> `(dmu, dsigma)`.
   The bandit reward is `dmu` standardized within the band (`bands.BandStats`).

9. **End of iteration: the LLM reads all rollouts, finds the dominant failure
   mode and its band, writes a new mutation prompt, and stores it in that band's
   pool.** `reflection.reflect` -> `prompt_bandit.PromptBandit.add_arm`. The pool
   (the set of bandit arms) grows over time.

## Running it

Frozen policy served by vLLM (or any OpenAI-compatible endpoint):

```bash
# serve the model first, e.g.
#   vllm serve Qwen/Qwen3-8B --port 8000
export LLM_API_KEY=EMPTY
python main.py --problem circle_packing
```

Offline wiring test with no endpoint (returns a stub program, so it exercises
the full loop without a real model):

```bash
python main.py --problem circle_packing --llm-backend dummy --num-iters 1
```

Config precedence is defaults in `config.py` < `configs/circle_packing.yaml` <
CLI flags (`--num-iters`, `--num-parents`, `--rollouts-per-parent`,
`--explore-eps`, `--seed`, `--llm-backend`).

## Design rationale (the non-obvious parts)

**Within-band delta normalization.** A raw mutation delta of +0.01 means
something completely different off a near_sota parent (tiny headroom) than off a
weak parent (large headroom). Feeding raw deltas to the prompt bandit would make
the near_sota pool look uniformly terrible and it would never get tuned. Each
band keeps a Welford running mean/std of its deltas and the bandit reward is the
z-score, so arms within a band are judged on that band's own scale. See
`bands.BandStats`.

**NIG, not Beta.** Mutation deltas can be negative, so the conjugate family is
Normal, not Bernoulli/Beta. `prompt_bandit.py` uses a Normal-Inverse-Gamma
posterior over each arm's (unknown mean, unknown variance) and Thompson-samples
the believed mean. A fresh reflection arm starts from the wide prior and gets
explored before the bandit commits.

**Sterile vs invalid, kept off the selectable set.** STERILE (ran but redundant)
and INVALID (broke a rule) children are stored separately from the parent pool.
Novelty is checked only against VALID states, so near-duplicates never pollute
the comparison set, and the worst-but-valid exemplar shown in a mutation prompt
is always a real valid program.

**The thing to watch: two non-stationary bandits feeding each other.** UCT is
estimating which parents are worth expanding while the prompt bandit is
simultaneously changing the conditional reward of every parent (the sampled
prompt determines child quality). So UCT's early statistics are computed over a
mutation operator that is still being tuned. Q = max child reward is the right
target for a discovery objective, but if this proves unstable, the levers are:
slow the reflection arm-growth, widen `uct_c`, or discount old UCT visit counts.

## Porting the other TTT problems

Only `circle_packing` is implemented. To add erdos / ac1 / ac2 / denoising /
gpu_mode, subclass `problems.base.Problem` and copy their `build_prompt` /
`score` / `seed_states` from the TTT codebase, then add two things: a
`build_seed_prompt` (usually `build_prompt(ParentContext())`) and a
`static_check` that surfaces the hard constraints they already enforce inline
(for example gpu_mode requires `@triton.jit` and forbids `identity`), so the
gate rejects them before wasting an evaluation. Their `preprocess` already
threads a seed where the entrypoint takes one, so the per-seed contract holds.
Register the subclass in `problems/registry.py`.
