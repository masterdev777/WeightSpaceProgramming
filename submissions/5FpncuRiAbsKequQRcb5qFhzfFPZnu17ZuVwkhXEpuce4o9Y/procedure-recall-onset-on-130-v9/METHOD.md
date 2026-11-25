# Procedure recall for maths and violation-onset credit for instructions, on the published model of 0009576150-0005 (#130)

This entry adds two layers, **R** (procedure recall for maths) and then **VO** (violation-onset credit for
instructions), to the published model of its declared parent `0009576150-0005` (#130, "own-close-carried-as-change";
`entry.json`), and uses that model as it is. R is derived in the run from the toolkit's practice sets and the model's
own generations: the model's thinking is started with one authored sentence that states a way of working; its own
right continuations, placed after the plain question, are distilled into 128 rewritten MLP units, so that it starts its
maths thinking that way unprompted, and nowhere else. VO then credits, inside the visible answer, the decision where
an instruction request first breaks (section 4). R is uploaded once built unless both of its screens show clear harm;
VO, fitted on the model that carries R, replaces that upload once it is exported in time. Every upload stacks on #130's
published edit. One driver, `derive/run_a.py`, runs every step and the uploads (section 5).

```
Fixed settings (derive/run_a.py and the scripts it calls; the run sets only --gpus)
start  : the declared parent's published edit, 0009576150-0005/edit/edit.json, as it is
R      : sentence 'concise'; 1024 practice maths problems x2; rows of 160 tokens, at most 2 per
         problem; keep rows on keepNM (383) + keepPZ (456), weight 3; 128 units outside every unit the start's chain
         rewrote, never the last layer; lr 1e-4, accum 8, S = max(4, 2 x training rows / 8)
R5     : screens math_hold and puzzle_hold paired with the start; R is held back only when BOTH are below the start
         by more than 2 se of the paired difference (clear harm); otherwise it is delivered (GO)
VO     : 1,920 requests (960 ifsynth + 960 tuluif) x4 + recombined requests (seeds 0-4) x2; word and sentence onsets;
         12 continuations per point; at most 2 pairs per point; beta 0.1, NLL 0.2; layers 4-21 with heads; 2 epochs;
         seeds 201-208, log-mean; strength 1.0; KEEP_W 0
minutes: left at the start of R2 >= 200; of R5 >= 90 for its screens (with less, R is delivered unscreened); of VO
         >= 205 (225 on the start alone); VO stopped at 25
other  : VO exported -> it is delivered (on top of R when R is GO; if its upload fails, R stands); VO skipped,
         failed or stopped -> R stands; R not GO -> VO runs on the start; neither layer -> nothing is uploaded
```

## 1. The start

The start is the edit that `0009576150-0005` (#130) published, read from the bundle's `0009576150-0005/edit/` together
with the chain it stands on. It is used as it is: none of its ops is changed, left out or applied a second time. R's
case starts from it (VO's too, on top of R when R is GO), and every upload names it as its parent, so the delivered
edit holds only this entry's own layers on top of it (R, R + VO, or VO alone when R is not GO).

What that model is, in our words (#130's METHOD describes it in full; its REPORT.md is in the bundle). #130 stands on
the champion of its time, `0009570269-0024` (#116, "procedure-workings"). It takes the published layer of
`0009569144-0010` (#111, "keep-the-answer-own-close": fixed rules - wrap-up units silenced, exit, break and commit
units, four direction removals on attention outputs - and a fitted round that favours answers which keep the result
they found and close their own thinking) and adds it to the champion as a change to the weights rather than as their
values, so that neurons touched by both layers keep both changes; the round goes in at half size and stays off the rows
the champion's own set ops write. On that merged model it fits one gain per MLP unit and per attention head that
prefers instruction answers keeping their request: the request-gain method of `0009552831-0015` as published by
`0009568726-0005` (#106). Its own practice checks decide what its final upload holds; the bundle holds that edit. The
champion of the moment is not used: the driver only names it in the log.

## 2. The failure R addresses, and why a recalled procedure

**The behaviour.** On the harder practice maths problems, the starts of this line that we explored work longer than
the problem needs: they check a step they have already checked, again and again, and run out of room, or keep going
after a result is checked instead of writing it in `\boxed{}`. #130's layers act on where and how the thinking
closes and keeps its result (the carried layer) and, if its published edit holds them, on the visible answer to
instruction requests (the request gains); R acts earlier, on how the maths working is set up from its first line.

**Why.** A way of working is missing, not knowledge: the same model gets more maths problems right when its thinking
starts from a short procedure in its own voice, "First, recall how to work efficiently: I keep the working short, do
not repeat a check I have already done, and once a result is checked I write it in \boxed{}." (`derive/recall_gen.py`,
`PREFIX['concise']`). The sentence states a general way of working; it names no question, option or hint. Forced on
other kinds of problem it costs, generated puzzles most, so the edit has to carry it on maths only: that is the job of
the keep rows (section 3, R4) and of the puzzle screen (R5).

**Practice evidence** (our exploration, practice splits only, each model paired with its start on the same problems
and seeds). Measured on the start this layer was first written for, the published edit `0009572333-0018`, unless
marked as the in-run rehearsal: that ran these steps on the champion `0009570269-0024` with the first twelve ops of
`0009569144-0010` (its fixed rules) placed on it, the twelve ops #130 carries at their full change. On this entry's
start, R5 checks for clear harm.
- *Forced recall, no fit* (two sets of 384 maths problems x2): +3.0 and +4.6 points with this sentence. A longer
  wording (restate what is asked, keep exact values, check against every condition) did about as well, 65.8% -> 69.3%
  (+3.5 +- 1.5) and 62.2% -> 65.5% (+3.3 +- 1.3), with its gain on the hard problems (level 4: 62% -> 68%, level 5:
  36% -> 42%) and fewer answers running out of room (5.6% -> 3.3%); the short sentence is used because it did as well
  and is shorter.
- *Forced on everything*: generated puzzles 89.1% -> 77.3%; multiple choice about the same (+0.4 / -3.3 on two sets of
  256); word problems about the same (-0.1 on 512); instruction requests lower (a constraint procedure there: -5.3).
- *The fit, first version* (concise): 1,213 fit rows from 677 problems and 384 keep rows (the non-maths answers; the
  456 puzzle keep rows came later, to keep the procedure off puzzles); held-out loss 1.21 -> 0.15.
- *In-run rehearsal* (64 units): forced recall right on 66.8% of 2,048 answers; 1,203 fit rows from 678 problems and
  839 keep rows; held-out loss 1.24 -> 0.157 (the 64- and 128-unit fits end alike there; this entry fits
  128). Its in-run screens: `math_hold` 66.1 -> 68.5 (+2.4 +- 1.1), `puzzle_hold` 89.3 -> 88.7 (-0.7 +- 0.8).
- *The fitted units* (concise, 64): maths (384 x4) +3.5 +- 1.1; generated puzzles about the same (+1.7 / +0.3 for two
  fits, 256 x4); the practice mix +0.0 (256 x4); 512 word problems +0.3. The edited model writes the procedure on none
  of the puzzles.
- *Did not work*: the start's own right maths answers distilled without the sentence (mixed hold-out -1.3 / -2.2,
  maths lower too); group-relative credit on its own maths answers (-0.5 to -2.2, more answers out of room); a
  procedure for instruction requests (-5.3) or for word problems (-1.2); units chosen without excluding the start's
  units (the fit overwrites part of the start).

**What sets it apart.** The fitted target is the model's continuation of an authored procedure, not a choice between
its own right and wrong answers: no pair, no preference and no answer of the model without the sentence is a target of
the fit (the start's own keep answers enter only as KL rows that hold it in place); the sentence is what carries it
(without it the same pipeline lowered the mixed hold-out, above). The recipe of forced recall, then distillation into
rewritten units, is that of `0009571995-0007` (#123) and `0009572069-0023` (#124), which used it to recall rules for
the generated puzzles; here the recalled text is one general way of working for maths, written for no particular kind
of problem, and the puzzle keep rows hold it off the puzzles. It leaves the start's layers in place: its units lie
outside every unit the start's chain rewrote, and each row it writes is the start's own row (any gain of the start
included) plus the fitted change. **What is changed:** the `gate_proj` and `up_proj` rows and the `down_proj` column
of 128 MLP units (`set_rows` / `set_cols`).

## 3. How the edit is found (R1-R5, as `derive/run_a.py` runs them)

`$S` = `0009576150-0005/edit/edit.json` (the driver resolves it to an absolute path); `R/` = `$WSP_WORK/recall/`;
`<runs>` = `$WSP_WORK/runs/mp`; `<start model>` = the path R1's build prints; `<gpus>` = the driver's GPU list
(`0,1,2,3`); minutes on 4 x RTX 4090.

- **R1** (20; mostly the start's build on the CPU). `python -m wsp.case init mp --profile qwen35-0.8b-think --contrast
  conclude --parent $S --practice gsm8k_train math_train mmlupro ifsynth bbhsynth --dev 512 --subset dev128=dev:128
  --subset dev256=dev:256 --subset holdout256=holdout:256`; the start's build (`case.model_path('mp', 'base')`);
  `python derive/recall_sets.py mp`, outside the case's dev and holdout splits: `recall_pool` (1024 `math_train`
  problems), `math_hold` (384 other maths problems, none with a pool problem's text), `keepNM` (383 not maths: 96 word
  problems, 128 instruction requests, 64 multiple choice, 5 puzzles of each of the 19 families), `keepPZ` (24 more
  puzzles per family, 456), `puzzle_hold` (10 per family, 190; never fitted or kept). Check: "R1 start built"; the
  five sizes.
- **R2** (40). Not started with fewer than 200 minutes left (R2-R4 take 80-105, R5 and the upload 35-55; section 5).
  Per GPU i of n, together:
  `CUDA_VISIBLE_DEVICES=i python derive/recall_gen.py mp base recall_pool R/RECALL.i.jsonl --n 2 --var concise --shard
  i/n`: the start's thinking is begun with the sentence, the model continues freely with the profile's sampling, and
  every answer is graded as usual. The shares are joined into `R/RECALL.jsonl`; `python derive/answer_kinds.py
  R/RECALL.jsonl` counts right, finished wrong and cut answers. Check: about two thirds right.
- **R3** (10). `python derive/prun.py mp base keepNM --n 1 --gpus <gpus>`, the same for `keepPZ`: the start's own
  answers. Check: "R3 keep answers: keepNM 383, keepPZ 456".
- **R4** (30-50; the start's screens set the pace).
  *Rows*: `python derive/recall_rows.py mp R/RECALL.jsonl R/ROWS --keep <runs>/base__keepNM.jsonl
  <runs>/base__keepPZ.jsonl --keep-w 3 --tokens 160 --per 2 --drop-mentions`. The right answers are placed after the
  PLAIN question (no sentence in the prompt); the target is the first 160 tokens of the forced answer, the sentence
  included, at most 2 per problem; answers that refer to "the hint", "the note" or "the given rule" are dropped; a
  tenth of the problems is held out to measure the fit. The keep answers, cut the same way, become keep rows, a KL to
  the unedited start with weight 3: they hold the rest of the model in place, the start's instruction answers and
  puzzles included.
  *Fit*, on the first GPU: `python derive/units_fit.py train <start model> R/ROWS.train.jsonl R/FIT --heldout
  R/ROWS.held.jsonl --rank 0 --units 128 --skip-last-mlp --exclude-units $S --lr 1e-4 --accum 8 --steps S --warmup 2
  --max-tokens 1536 --eval-every 20 --log-every 10`, S = max(4, 2 x training rows / 8). The 128 units are those with
  the largest loss gradient on the rows, outside the last layer and outside every unit that the start's chain rewrote
  (`--exclude-units` reads `$S` and every link under it: units written by `set_rows` / `set_cols`, or by a `dir_ablate`
  restricted to their rows or columns). Beside the fit, on the other GPUs, the start's two screens one after the
  other: `prun.py mp base math_hold --n 4`, `puzzle_hold --n 4`.
  *Export*: `python derive/units_fit.py export R/FIT mp PROC128 --model <start model>` (each written row and column is
  the start's own plus the fitted change), the toolkit's model digest, the build of `PROC128` on the CPU. The driver
  then initialises VO's case `kr` on `PROC128` (`--practice gsm8k_train math_train ifsynth tuluif --dev 128 --holdout
  128`) and builds its start on the CPU while R is screened; VO uses it only if R is GO (section 4).
  Check: about 1,200 fit rows from about 680 problems and about 840 keep rows; "excluded units of the parent chain: N
  scores zeroed" with N > 0; the held-out loss falls from above 1 to below 0.2.
- **R5** (15-30). `prun.py mp PROC128 math_hold --n 4 --gpus <gpus>`, then `puzzle_hold --n 4`, each followed by
  `python derive/grid.py table mp <set>`: the toolkit's paired result against the start on the same problems and
  seeds. A screen's standard error is about 1 point (384 problems x4), as large as the shifts the screens are asked
  to tell apart, so the screens do not decide by their sign: R is held back only when BOTH screens are
  below the start by more than 2 standard errors of the paired difference (a fit that clearly broke something);
  otherwise `python -m wsp.case freeze mp PROC128` and R is delivered. With fewer than 90 minutes left the screens
  are skipped and R is delivered as built, unscreened and not frozen (`wsp.case freeze` takes only an edit with results). Check: "R5 PROC128 vs the start: math_hold ..., puzzle_hold
  ... -> no clear harm: frozen PROC128; R = GO".
- **Upload** (about 20, on the CPU), as soon as R is GO, while VO starts. The report `$WSP_WORK/runa/REPORT_R.md` (the
  start; R's forced-recall share, rows, fit line and screen tables; a line that VO may follow and, once exported,
  replaces this upload; `wsp.upload` adds the case's results tables); `python -m wsp.upload mp PROC128 --report
  $WSP_WORK/runa/REPORT_R.md --task task.json --dry-run` (the package); the driver's check that the package names
  `0009576150-0005` as its parent and carries exactly R's ops; `axonry-check $WSP_WORK/deliver/mp__PROC128/edit --task
  task.json`; the same upload command without `--dry-run`. Check: "package mp/PROC128 ok: parent 0009576150-0005, ...
  stacks on 0009576150-0005"; "axonry-check mp/PROC128: passed"; "UPLOADED mp/PROC128".

## 4. VO: violation-onset credit for instructions

**The behaviour.** The start follows most practice instruction requests. When it fails one, the request's own
checker (the toolkit's instruction checkers, as `wsp.grade` uses them) tells three shapes apart on the visible answer:
- *growing*: once broken, more text cannot repair it (a comma under "no commas", a capital under "all lowercase", a
  forbidden word, the word that crosses a "less than" limit, a keyword once too often): the answer was right up to
  one word;
- *missing*: more text could repair it (postscript, end phrase, sections, placeholders, "at least" counts);
- *start*: decided by how the answer begins (JSON, repeat the request, quotation, numbered bullets, the first word of
  a paragraph, two responses).

**Why it happens (suspected).** The start keeps most requests; a growing failure is one choice of its usual way of
writing (a comma, a capital, a word it likes, one sentence too many), taken where the request is far back in its
context: the answer was right up to that word.

**Why a local fix.** For a growing failure everything before the onset was fine; the loss is one decision. Credit
spread over the whole answer is spent on hundreds of tokens that were right, and resampling the whole answer mostly
changes how it is written, not that decision.

**How the edit is found** (`python derive/run_onset.py --gpus <gpus> --case <kr or ku> --keep-w 0 --no-upload --k 12
--req-sets 5`, launched by the driver; each step skips what an earlier run finished).
1. *The start's answers.* 4 answers to 1,920 practice instruction requests outside the case's dev and holdout splits
   (960 from `ifsynth` and 960 from `tuluif`, each after a fixed shuffle, so that the mix does not follow the sets'
   sizes; 12,000 tokens), and 2 answers to each of the requests recombined from the practice material by #94's
   `mkreq.py` (its own request kinds, none added; seeds 0-4; about 6,600 requests), which fail more often and more
   variously. `mkreq.py` takes the bodies of its requests from outside every set of the case.
2. *Onsets* (`derive/if_onset.py`). For a failed, finished answer each failing instruction is classed. For a growing
   one the checker runs on growing prefixes of the visible answer (word boundaries, binary search, valid because a
   growing failure stays broken; the two case instructions use the first letter of the wrong case): the onset is the
   start of the word whose inclusion first breaks it. For a missing one it is the start of the last paragraph (of the
   last sentence when the answer is one paragraph); an answer with both takes its earliest growing onset. Start
   failures are left out, and so is any branch point at the very start of the visible answer (it would only resample
   the whole answer after the thinking, as #107's branch kind `if` does). The visible answer is the text after the
   last `</think>`, as the grader reads it. Each onset gives two branch points: just before the breaking word, and the
   start of its sentence or line, one decision earlier.
3. *Continuations* (`derive/onset_branch.py`, the branch kind `onset`). Prompt = the chat prompt plus the answer's own
   text up to the branch point (thinking, `</think>`, the visible answer so far); 12 continuations with the profile's
   sampler (the prefix's presence penalty as a fixed logit bias); each graded by the request's checker on the full
   answer. The original answer counts as one more failure of its point.
4. *Pairs and fit.* At each point a continuation that keeps the request against one that breaks it (or the
   original), at most 2 pairs per point, cut continuations counted as failures. The fitter is #107's published
   `pref_gains2.py`, unchanged: DPO-style preference (beta 0.1) plus 0.2 x NLL of the kept continuation, one gain per
   MLP unit and per attention head of layers 4-21, 2 epochs; the loss covers the continuation only (the prefix is
   context); it runs without its keep term (`KEEP_W` = 0). Eight fits (seeds 201-208, rounds of one per GPU),
   averaged in log space: single fits move different units, and their average is what keeps the other tasks in place.
5. *Strength* fixed at 1.0 (the merged gains as fitted), exported with #107's `fit_gains.py export` and built. It is
   not chosen in the run: a screen of a few hundred held-out requests cannot separate strengths a point apart.
6. *Check.* VO has no paired screen of its own in this run: after R's steps, VO's 166 minutes and the upload, the
   6 hours leave no room for one (two models on a few hundred held-out requests and puzzles x4 take about 35
   minutes). What it measures on practice data is printed on the way: the start's answers and how many fail, the
   onsets by class and the pairs the fits use (VO's status log `$WSP_WORK/onset/status.log` and REPORT.md); per
   branch point how many of the 12 continuations end right (finished, at least 20 tokens and half the original
   answer's length) in `$WSP_WORK/onset/br_<i>.log`; each fit's loss and preference accuracy on its pairs in
   `$WSP_WORK/onset/fit_<seed>.log`. If any VO step fails, or VO is stopped at the deadline guard, VO is not delivered
   and R stands. The delivered REPORT.md carries the onset and pair lines and VO's step lines.

**What is changed.** Per-unit gains (`scale_rows` on `up_proj`) and per-head gains (`scale_cols` on the attention
output projections) of layers 4-21, about 0.4 MB.

**What sets it apart.** The branch kind `onset` (this team's #138, `0009578577-0019`) branches where the request's own checker, run on growing prefixes
of the visible answer, first finds it broken, and one sentence earlier; the loss covers only the continuation; the
fitter is #107's published code, unchanged. The material is local (a kept prefix and the model's own continuations
from it), not whole answers and not answers regenerated after the thinking. No answer is mended, cut or rewritten by
rule: at both branch points the preferred side is one of the model's own continuations, graded by the request's
checker (unlike the answers repaired by rule of #96, `0009565562-0012`).

**Order.** VO is fitted on the model that already carries R (its case `kr` starts from R's exported spec), so its
pairs are the answers of the model that will be delivered; when R is not GO it is fitted on the start itself (case
`ku`). The parent's own instruction layer (its request-gain round, which resamples whole answers after the thinking)
stays as it is; VO adds branch points inside the visible answer.

## 5. Running it (4 x RTX 4090, 6 hours)

**Copy** (agent machine, from the unpacked bundle; each start directory under its id, next to task.json):
```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r 0009576150-0005 sandbox:/workspace/work/      # the declared parent: the start
```
task.json holds the upload credentials: it stays out of PROGRESS.md, notes and logs.

**Launch** (sandbox; it sets `WSP_CASES`, `WSP_WORK`, `PYTHONPATH`, otherwise
`export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp`):
```
cd /workspace/work && mkdir -p "$WSP_WORK" "$WSP_CASES"
setsid nohup python derive/run_a.py --gpus 0,1,2,3 > "$WSP_WORK/run_a.out" 2>&1 < /dev/null &
tail -n 5 "$WSP_WORK/runa/status.log"
```
The driver writes one line per step to `$WSP_WORK/runa/status.log` and its decisions to `$WSP_WORK/runa/state.json`;
started again (same `--gpus`), it skips what is done and never decides twice. It refuses to start while another
driver runs ("run_a.py already runs"); whether one runs: `kill -0 "$(cat $WSP_WORK/runa/run_a.pid)" && echo running`.
Each R step's log is in `$WSP_WORK/recall/`, VO's in `$WSP_WORK/onset/`, the uploads' in `$WSP_WORK/runa/`. Every ~10
minutes read the last lines of the status log and make the first line of `/workspace/work/PROGRESS.md` one plain
sentence of at most 20 words on where the run is (the status lines go below it). On "continue", read PROGRESS.md and
the status log and check whether the driver runs before anything else. A build prints nothing for up to ~20 minutes
(python at ~100% CPU): not a hang, never kill it. `--no-upload` runs every step and only packages and verifies (give it
its own `WSP_WORK` and `WSP_CASES`: a later run in the same directories finds every decision taken);
`--smoke` is a rehearsal at tiny sizes in separate directories. The run ends with "DONE: ..." in the status log; then
`axonry-done`.

**Time plan** (per step, from runs of each component on 4 x RTX 4090: in-run rehearsals of the R steps on related
starts took about 2 h 05 to the end of the screens, VO's steps 166 minutes). R takes about 2 hours, VO about 2 h 50.
VO starts only with at least 205 minutes left (its ~170, up to 10 to package and check it, the upload's 25), that is
when R5 ends by about 2:35; R5 ends at about 2:05-2:25, which leaves 10-30 minutes in hand. When R5 ends later, VO is
skipped, R's upload is the delivered edit and about 3 hours are left in reserve. A VO that runs late is stopped with 25
minutes left, and R's upload stands.

| from | to | step |
|---|---|---|
| 0:00 | 0:10 | unpack, copy, launch |
| 0:10 | 0:30 | R1: case on the start, the start's build (CPU), the sets |
| 0:30 | 1:10 | R2 forced recall (40, up to 45) |
| 1:10 | 1:20 | R3 keep answers (10) |
| 1:20 | 1:50-2:10 | R4 rows; fit (~17 min for ~510 steps), export, build on GPU 0 and the CPU, beside the start's two screens on GPUs 1-3 (2,296 answers: they set the pace); VO's case built on the CPU |
| 1:50-2:10 | 2:05-2:25 | R5 screens, harm check, freeze |
| 2:05-2:25 | 2:25-2:45 | R packaged, checked and uploaded on the CPU while VO starts |
| 2:05-2:25 | 4:55-5:15 | VO: answers 23, recombined 37, onsets + continuations 27, 8 fits 69, export and build 10 |
| 4:55-5:15 | 5:15-5:35 | final report, package, check, upload of VO |
| 5:15-5:35 | 6:00 | reserve |

## 6. Guards and failures (all in the driver; minutes left are read from `task.json` -> `run.deadline`)

- **The start missing**: if task.json does not list `0009576150-0005` among its starts, or its directory was not
  copied, the driver stops at once and says which; copy the directory and start the driver again. No other start is
  used.
- **R not delivered.** R is skipped with fewer than 200 minutes left before the forced recall, held back when R5 finds
  clear harm on both screens, or failed (any command or error in R1-R5); with fewer than 90 minutes left at R5 it is
  not screened but delivered as built. When R is not delivered, VO runs on the start itself (case `ku`) if at least
  225 minutes are left (205, plus up to 20 for the build of its case; after a skip that much is never left), and is
  delivered once exported; if not, nothing is uploaded. R's decision is kept in `state.json`; a restarted driver
  keeps it and resumes where it stopped.
- **The upload.** It starts as soon as R is GO. `axonry-check` runs when the sandbox has it and at least 25 minutes are
  left; with less, a package is sent without it only when no other upload stands (VO's is then not sent, and R's
  upload stands). Each send is tried up to three times, 30 s apart. If R's delivery fails, the final step tries it
  once more. If VO's package, check or send fails, R's upload stands as the delivered edit (sent again first, up to
  six tries, when VO's send may have overwritten its files), the run ends with "DONE", and VO is not tried again. If
  nothing gets through, the status log says "final: no candidate was delivered", and the driver started again after
  the cause is fixed delivers again, VO first (R's GO and VO's export stand).
- **VO.** It is not started with fewer than 205 minutes left (225 on the start alone); a driver started again counts
  only VO's steps still to do, plus the upload. It is stopped when fewer than 25 minutes remain (run_onset.py and
  every job under it); R's upload then stands (on the start alone nothing is uploaded). A VO step that fails stops VO
  (R stands); a driver started again stops any run_onset.py a killed driver left behind, takes VO up from its last
  finished step, and delivers a VO already exported. While R's upload stands, VO's upload is not started with fewer
  than 25 minutes left, not sent unchecked when the sandbox has `axonry-check` (the check needs 25 minutes left once
  VO is packaged), and not sent with fewer than 8 (a cut upload could spoil R's).

## 7. Credits and files

- **The start**: `0009576150-0005` (#130, "own-close-carried-as-change", published), used as it is. By its METHOD it
  rests on `0009570269-0024` (#116, the champion of its time), the published layer of `0009569144-0010` (#111) added
  as a change, and the request gains of `0009552831-0015` fitted with the scripts and steps of `0009568726-0005` (#106);
  its way of merging follows task arithmetic (Ilharco et al., 2022), with TIES-merging (Yadav et al., 2023) on where
  merged changes interfere. Nothing of these is re-derived or changed here.
- **R**: the forced-recall-then-distil recipe and `prun.py`, `grid.py`, `answer_kinds.py`, `recall_rows.py`,
  `units_fit.py` were first published in `0009571995-0007` (#123, recall-the-rule) and `0009572069-0023` (#124,
  rule-recall-on-own-close), by a member of this team. The maths procedure layer itself (the concise sentence,
  `recall_sets.py`, `recall_gen.py`, the 128 units, the keep rows and the two practice screens) is that of this team's
  finished entry `0009578737-0026` (#139, the same layer on #130 with an in-run GO rule), unchanged; `0009578522-0013`
  (#136, by this team) ran the first procedure recall on #130 (combined sentence, 64 units) with the same pipeline;
  what changes in R here is its delivery rule (R5: held back only on clear harm on both screens). Choosing units outside
  those already rewritten is the practice of `--exclude-neurons-from` in `train_sparse.py` of `0009550031-0005`,
  extended to every link of the start's chain as in the `units_fit.py` of #136 and #139.
- **VO**: `0009568722-0012` (early-decision credit, where these files first appear), #107 (`0009568831-0002`) and #106 (`0009568726-0005`), which publish the same files byte for byte, for
  `branch2.py` (`onset_branch.py` = it plus the kind `onset`), `pref_gains2.py`, `fit_gains.py`, `merge_gains.py` and
  `shard_gen.py`; #94 (`0009559255-0006`) for `mkreq.py` and `genreq.py`, both byte for byte as #94 published them,
  and the item-index patch in `with_req.py`; #96 (`0009565562-0012`) as the closest prior idea (a failed answer
  repaired by rule and preferred to itself). The hang guards in `prun.py`, `shard_gen.py`, `run_onset.py` and
  `run_a.py` are this team's (aldiendix's watchdog). VO itself is the layer of this team's finished entry `0009578577-0019` (#138, violation-onset-on-130): its onset classes and two branch points, the kind `onset`, `if_onset.py`, `run_onset.py` and every VO setting above are #138's, unchanged (`onset_branch.py` and `if_onset.py` differ only in their docstrings; `run_onset.py` here stops after the export and build). What changes is where it is fitted: on the start + R (case `kr`). `run_onset.py` exports
  and builds VO; the driver reports, packages, checks and uploads.
- `run_a.py` (the driver: the start, R1-R5, VO's case and launch, the reports, the package check, the uploads): ours.

## 8. Data for the run

Use only the toolkit's `practice:` sets, the bundle's start directories and the models' own generations; never load a
`bench:` set, and download nothing. The start's arrays are read from its published directory; every array of R and VO
is fitted in the run.
