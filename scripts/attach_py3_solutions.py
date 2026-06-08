"""
Attach a correct Python 3 solution to each problem in Gen-Verse/CodeContests (test split).

Strategy (in order):
  1. Use an existing Python 3 solution from the original deepmind/code_contests dataset.
  2. Convert a Python 2 solution using 2to3, then validate against example I/O.
  3. Generate a solution via LLM (gpt-4o-mini), then validate against example I/O.

Output: JSON lines file where each record has task_id + py3_solution + source.
"""

import os
import json
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset
import openai

load_dotenv(Path(__file__).parent.parent / ".env")
client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

MODEL = "gpt-5.4-mini-2026-03-17"
TIMEOUT_SECS = 10  # per test-case execution


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_solution(code: str, stdin_input: str, timeout: int = TIMEOUT_SECS) -> tuple[bool, str]:
    """Run Python 3 code with given stdin. Returns (success, stdout_or_error)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        result = subprocess.run(
            ["python3", fname],
            input=stdin_input,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return False, result.stderr
        return True, result.stdout
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    finally:
        os.unlink(fname)


def outputs_match(got: str, expected: str) -> bool:
    return got.strip() == expected.strip()


def validate_on_examples(code: str, example_inputs: list[str], example_outputs: list[str]) -> bool:
    """Return True if code produces correct output on all example test cases."""
    if not example_inputs:
        return True  # no examples to validate against
    for inp, exp in zip(example_inputs, example_outputs):
        ok, out = run_solution(code, inp)
        if not ok or not outputs_match(out, exp):
            return False
    return True


def py2_to_py3(code: str) -> str:
    """Run 2to3 on code string and return converted code."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        subprocess.run(
            ["2to3", "-w", "-n", fname],
            capture_output=True,
            text=True,
        )
        return Path(fname).read_text()
    finally:
        os.unlink(fname)


def llm_generate(problem_text: str, example_inputs: list[str], example_outputs: list[str]) -> str:
    """Ask the LLM to write a Python 3 solution."""
    examples_block = ""
    for i, (inp, out) in enumerate(zip(example_inputs, example_outputs)):
        examples_block += f"\nExample {i+1} Input:\n{inp}\nExample {i+1} Output:\n{out}"

    prompt = textwrap.dedent(f"""
        Solve the following competitive programming problem in Python 3.
        Output ONLY the Python 3 code, no explanation, no markdown fences.

        Problem:
        {problem_text}

        {examples_block}
    """).strip()

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    code = resp.choices[0].message.content.strip()
    # Strip markdown fences if model added them
    if code.startswith("```"):
        lines = code.splitlines()
        code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return code


# ---------------------------------------------------------------------------
# Build lookup: description_prefix -> list of solutions per language
# ---------------------------------------------------------------------------

def build_lookup():
    print("Loading original deepmind/code_contests ...")
    lookup = {}  # description[:120] -> {"py3": [...], "py2": [...]}
    for split in ("train", "valid", "test"):
        ds = load_dataset("deepmind/code_contests", split=split)
        for ex in ds:
            key = ex["description"][:120]
            py3 = [s for l, s in zip(ex["solutions"]["language"], ex["solutions"]["solution"]) if l == 3]
            py2 = [s for l, s in zip(ex["solutions"]["language"], ex["solutions"]["solution"]) if l == 1]
            lookup[key] = {"py3": py3, "py2": py2}
    print(f"  Lookup built: {len(lookup)} problems")
    return lookup


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_split(gv_split: str, output_path: str):
    lookup = build_lookup()

    print(f"\nLoading Gen-Verse/CodeContests split={gv_split} ...")
    gv_ds = load_dataset(
        "Gen-Verse/CodeContests" if gv_split == "test" else "Gen-Verse/CodeContests_train",
        split=gv_split,
    )
    print(f"  {len(gv_ds)} problems")

    stats = {"py3_direct": 0, "py2_converted": 0, "llm": 0, "failed": 0}
    results = []

    for i, ex in enumerate(gv_ds):
        task_id = ex["task_id"]
        question = ex["question"]
        example_inputs = ex["example_input"]
        example_outputs = ex["example_output"]
        key = question[:120]

        sol = None
        source = None

        orig = lookup.get(key, {"py3": [], "py2": []})

        # --- Strategy 1: existing Python 3 — take first, assume correct ---
        if orig["py3"]:
            sol = orig["py3"][0]
            source = "py3_direct"

        # --- Strategy 2: 2to3 on Python 2 ---
        if sol is None:
            for candidate in orig["py2"]:
                converted = py2_to_py3(candidate)
                if validate_on_examples(converted, example_inputs, example_outputs):
                    sol = converted
                    source = "py2_converted"
                    break

        # --- Strategy 3: LLM generation ---
        if sol is None:
            for attempt in range(3):
                try:
                    generated = llm_generate(question, example_inputs, example_outputs)
                    if validate_on_examples(generated, example_inputs, example_outputs):
                        sol = generated
                        source = "llm"
                        break
                    time.sleep(1)
                except Exception as e:
                    print(f"    LLM error on task {task_id}: {e}")
                    time.sleep(2)

        if sol is None:
            source = "failed"
            stats["failed"] += 1
        else:
            stats[source] += 1

        results.append({"task_id": task_id, "source": source, "py3_solution": sol})

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(gv_ds)}] {stats}")

    # Write output
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nDone. Stats: {stats}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--output", default="py3_solutions_{split}.jsonl")
    args = parser.parse_args()

    out = args.output.replace("{split}", args.split)
    process_split(args.split, out)
