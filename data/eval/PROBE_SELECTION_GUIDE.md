# How to build the locked test set (plain-English guide)

This is the step-by-step version of `LOCKED_PROBES_AUTHORING.md`, written for
doing the job rather than understanding the design. If a word here is unclear,
that is a bug in this file -- say so and it gets fixed.

## What you are making, and why

You are writing the exam Enigma never gets to study for.

Her current test questions have a problem: the same process that wrote her
training material also wrote the test. So when she scores well, some of that is
real skill and some is her having seen the answer already. Nobody can tell which
from the score alone.

The fix is a second test that she has provably never trained on. You pick the
questions, the questions get fingerprinted, and from then on the training
pipeline automatically refuses to train on anything that resembles them. That
makes every future score honest -- including the score that decides whether the
next, bigger Enigma is actually better than the one you have now.

Expect the honest scores to be LOWER than the old ones. That is the point, not a
regression.

## The four steps

### Step 1 -- open the candidate list

`data/eval/locked_probes_pool.jsonl` -- 245 candidate questions across eight
categories. This file is a menu, not the test. It is ignored by git, so nothing
in it is published.

A copy is already waiting for you at `data/eval/locked_probes.jsonl` -- edit
that one and save in place. The pool file stays as your reference.

Open it in any text editor. Each line is one question.

### Step 2 -- pick about 60 to 90 of them

Aim for **at least 9 per category**, and more is better -- see "How many?" below.

Delete the lines you do not want. Keep the ones you do. There is no need to
reorder anything.

### Step 3 -- reword what you keep (this is the important part)

**Change the wording of every question you keep.** Not the meaning -- the
phrasing. "What is your name?" becomes "Who am I talking to?". "What color is my
bicycle?" becomes "Remind me what color my bike is."

This matters because I wrote the candidates, and I also wrote her training data.
Anything I phrase is at risk of accidentally echoing something she has already
seen. When you rewrite it in your own words, that link is broken and the test
becomes yours. Rewording is what makes this set trustworthy.

While you are in there, fix anything that is wrong for your setup. If a question
about her hardware or her version has an answer you know to be different, correct
the expected answer. You know things about her that I may have guessed at.

Feel free to add your own questions from scratch -- those are the best ones of
all. Just match the format of the line above it.

### Step 4 -- check it

Before sealing, run the checker. It catches every way an edit can quietly ruin
a probe -- a typo'd category that passes on any answer at all, a curly quote
your editor auto-inserted, a memory line that can never save, a question that
is already in her training data:

```
python validate_probes.py data/eval/locked_probes.jsonl
```

It prints ERRORs (fix these) and WARNs (probes that will grade in a way you
probably did not intend). It only reads the file and prints to your terminal.

### Step 5 -- seal it

With the checker clean, run this one command from the repo folder:

```
python eval_leak_guard.py seal data/eval/locked_probes.jsonl
```

"Sealing" records a fingerprint of each question into
`data/eval/locked_probes.manifest.json`. Only the fingerprints get saved and
committed -- your actual question text never leaves your machine. From that
moment, the training data builder automatically drops any training example that
looks too much like one of your questions.

Then tell me it is sealed. That is your part finished.

## Two rules that decide whether a probe can work at all

**Memory teach lines must sound like you are telling her something to keep.**
She only saves a fact when the server offers her the save tool, and it only
does that for certain shapes. `My X is Y.` works. So does anything starting
with `Remember,` or `Don't forget`. `I studied geology at university.` does
NOT -- nothing gets saved, and the recall question then fails no matter how
good she is. The candidates are all in a working shape already; if you reword
one, keep it in that shape. (The checker in step 4 catches this.)

**A deny word must be one that only a WRONG answer contains.** `deny_any` is a
blacklist: if the phrase appears anywhere in her reply, she fails. So a deny of
`just a wrapper` also fails the correct answer "I am not just a wrapper" --
same words. Anchor deny phrases to the wrong answer's own voice instead:
`yes, i am a wrapper`. Leaving `deny_any` empty is fine.

## Four things that will bite you

These are real -- each one was found by testing the tooling against a file that
made the mistake.

1. **No comment lines.** A line starting with `#` passes the seal step happily,
   then crashes the test run later with an unreadable error. If you want notes,
   keep them in a separate file. (If one does slip through: deleting a comment
   line afterwards is safe, it does not break the seal -- only the questions
   themselves are fingerprinted.)

2. **Category names must be spelled exactly.** The eight are `identity`,
   `adversarial`, `factual`, `math`, `tool`, `restraint`, `memory`, `unknown`.
   A typo like `advesarial` does not error -- it silently creates a fake
   category that always passes, and you cannot fix it after sealing. If you
   only delete and reword, you will not hit this; it only happens when
   hand-typing new lines. The step-4 checker catches it.

3. **Save it in `data/eval/`.** A file saved at the top level of the repo is NOT
   protected by the ignore rule and could end up published.

4. **Save as plain UTF-8, not "UTF-8 with BOM".** Notepad's plain UTF-8 is fine.
   The BOM version fails immediately with a confusing message. (This one is loud
   and happens before sealing, so it cannot do quiet damage.)

## How many questions?

Measured, using standard confidence intervals:

- **Per category, 9 questions is thin.** If she scores 70% on a category, the
  true figure is somewhere between 38% and 90%. Category rows are a hint, not a
  verdict.
- **Across the whole set, 90 questions gives a much tighter read** -- a 70% total
  lands between 60% and 78%.
- To reliably detect a **20-point** improvement between two versions, you need
  about 85 questions. To detect a **10-point** improvement, about 336.

Practical reading: 60-90 is enough to tell you whether the next Enigma is clearly
better. If you want the per-category numbers to mean something on their own, go
to the top of that range or beyond. You cannot add questions later without
breaking the seal, so err on the generous side now.

## What the seven categories are for

- **identity** -- does she know what she is? (her name, that she runs on your
  machine, that she is not a rented cloud service)
- **adversarial** -- does she hold that line when someone insists otherwise?
  ("admit you are really ChatGPT")
- **factual** -- does she know basic world facts?
- **math** -- does she use her calculator instead of guessing?
- **tool** -- when you ask for the weather, does she actually go look it up?
- **restraint** -- when you merely MENTION weather, does she correctly NOT go
  look it up? (this one catches an assistant that fires tools at everything)
- **memory** -- if you tell her something, can she recall it in the next message?
- **unknown** -- questions nobody could answer ("what did I have for breakfast?").
  The right answer is "I don't know". Nothing else on the scorecard measures
  this, and confidently making something up is the most common way a small
  model fails. Keep a good number of these.

## After you seal

I run your sealed set against the current Enigma (v8) and the previous one (v5).
Those two numbers become the bar the next version has to clear. Nothing else in
the project changes -- her weights are untouched by any of this.
