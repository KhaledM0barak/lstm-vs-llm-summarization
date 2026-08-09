# Recovering after a Terminal restart

## 1. The background jobs are already safe

The LLM sweep is **not** a child of Terminal. It was launched with `nohup` and has
been reparented to `launchd` (PPID 1), so quitting and reopening Terminal will
not kill it:

```
PID 63922  PPID 1   bash scripts/finish.sh
PID 64090  PPID 63922  python -m src.llm.baseline ...
```

Verify it's still alive after restarting:

```bash
pgrep -fl "src.llm.baseline|finish.sh"
```

Check progress:

```bash
cd ~/lstm-vs-llm-summarization
wc -l runs/llm/*.jsonl          # each setting counts up to 500
tail -c 300 finish.log | tr '\r' '\n' | tail -2
```

**Even if it does die, nothing is lost.** The runner is resumable — it reads the
existing output file, skips every example already completed, and continues:

```bash
cd ~/lstm-vs-llm-summarization
bash scripts/finish.sh
```

Re-running is always safe. It never redoes finished work.

## 2. Recovering the Claude Code conversations

Conversations are stored on disk as JSONL under `~/.claude/projects/`, keyed by
the directory Claude Code was started in. They are written continuously, so
nothing is lost when Terminal quits.

**Sessions started in the home directory** (this project's conversation):

```bash
cd ~
claude --resume
```

That opens a picker listing recent conversations — choose from the list.

To jump straight back into the most recent one instead of picking:

```bash
cd ~
claude --continue
```

**Sessions started elsewhere** — `--resume` only lists conversations for the
directory you're in, so `cd` there first:

```bash
cd ~/Desktop
claude --resume
```

### The important detail

`--resume` and `--continue` are **directory-scoped**. If the picker looks empty
or is missing the conversation you want, you are in the wrong directory. The two
directories with saved sessions on this machine are:

- `/Users/kmac` ← the project conversation
- `/Users/kmac/Desktop`

### This conversation's ID

```
1d4da1e9-7f7b-4f84-92c2-214d3265985b
```

Transcript: `~/.claude/projects/-Users-kmac/1d4da1e9-7f7b-4f84-92c2-214d3265985b.jsonl`

## 3. Order of operations

1. Note the session IDs you care about (or just remember the directories).
2. Change the privacy setting; quit and reopen Terminal.
3. `cd ~ && claude --resume` → pick this conversation.
4. `pgrep -fl src.llm.baseline` to confirm the sweep survived.
5. If it didn't: `cd ~/lstm-vs-llm-summarization && bash scripts/finish.sh`.

## 4. What is already committed and pushed

Everything except the in-progress LLM outputs is on GitHub:
<https://github.com/KhaledM0barak/lstm-vs-llm-summarization>

Trained checkpoints are local only (gitignored, ~59 MB each) but they are on disk
in `runs/*/best.pt` and do not need re-training under any circumstances.
