"""Multi-turn iterative refinement eval for CodeContest problems.

Pipeline per problem:
  Turn 0  : solver generates code_0 from the problem statement alone.
  Turn t>0: tester sees (problem + code_{t-1}) and generates K targeted test
             cases in a single response; solver sees the test results as a new
             conversation turn and refines its code.

The module is self-contained (stdlib only) and receives the tunix sampler and
tokenizer as arguments, so it can be imported without JAX being installed.
"""

import ctypes
import io
import re
import sys
import threading
import typing
from typing import Dict, List, Optional, Tuple

# ── Prompt templates ──────────────────────────────────────────────────────────

CODE_PROMPT_TEMPLATE = (
    "<|im_start|>You are a helpful assistant help user solve problems. "
    "<|im_end|>\n<|im_start|>User: "
    "You need to think first then write python script. "
    "You should use input() to input and print() to output in your script. "
    "Your code should output the results based on the input read in, rather than generating the given test example.\n"
    "This is the problem:\n{problem} "
    "<|im_end|>\n<|im_start|>Assistant: "
)

TESTER_TARGETED_TEMPLATE = (
    "<|im_start|>You are a helpful assistant that reviews code and generates targeted test cases. "
    "<|im_end|>\n<|im_start|>User: "
    "Given a coding problem and a candidate solution, generate {k_case} test cases to evaluate the solution.\n\n"
    "This is the problem:\n{problem}\n\n"
    "This is the candidate solution:\n```python\n{code}\n```\n\n"
    "Study the code carefully. If you spot logical errors, edge cases it might mishandle, or inputs "
    "that could produce wrong results, generate test cases designed to expose those weaknesses. "
    "If you believe the code is correct, generate test cases that confirm it handles important cases correctly.\n"
    "For each test case, think step by step: design the input, trace through the code to predict its "
    "output, then independently compute the correct expected output. If unsure, revise the input.\n"
    "You MUST provide exactly {k_case} test cases, each in the following format:\n\n"
    "**Test Input:**\n```input here```\n\n"
    "**Test Output:**\n```output here```\n\n"
    "**Explanation:**\n\nexplanation here.\n\n"
    "(repeat for each test case)\n "
    "<|im_end|>\n<|im_start|>Assistant: "
)

# User-turn message only — stitched into multi-turn conversation by build_refine_prompt.
SOLVER_REFINE_FEEDBACK_TEMPLATE = (
    "The following test cases were run against your code. "
    "Note: the expected outputs were generated automatically and may not be correct — "
    "use your own judgment when deciding whether your code has a bug.\n\n"
    "{feedback_block}"
    "Please review your solution. If you believe your code is correct and the expected outputs "
    "are wrong, you may keep it unchanged. Otherwise, identify the bug and write an improved solution.\n"
)

# ── Utilities (verbatim from notebook cells) ──────────────────────────────────

_exec_lock = threading.Lock()


def _kill_thread(t: threading.Thread) -> None:
    if t.ident is None:
        return
    ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(t.ident),
        ctypes.py_object(SystemExit),
    )


def run_code(code, stdin_str, timeout=5.0):
    # type: (str, str, float) -> str
    result: list[str | None] = [None]
    buf = io.StringIO()

    def target() -> None:
        lines_iter = iter(stdin_str.splitlines())

        def fake_input(prompt: str = "") -> str:
            try:
                return next(lines_iter)
            except StopIteration:
                raise EOFError

        ctx: dict = {
            "__name__": "__main__",
            "input": fake_input,
            "List":     typing.List,
            "Tuple":    typing.Tuple,
            "Optional": typing.Optional,
        }
        try:
            exec(compile(code, "<solution>", "exec"), ctx)  # noqa: S102
            result[0] = buf.getvalue()
        except SystemExit:
            result[0] = buf.getvalue()
        except Exception as exc:
            result[0] = f"error: {exc}"

    with _exec_lock:
        saved_stdout, saved_stdin = sys.stdout, sys.stdin
        sys.stdout = buf
        sys.stdin  = io.StringIO(stdin_str)
        try:
            t = threading.Thread(target=target, daemon=True)
            t.start()
            t.join(timeout=timeout)
        finally:
            sys.stdout = saved_stdout
            sys.stdin  = saved_stdin

    if t.is_alive():
        _kill_thread(t)
        return "timeout"
    return result[0] if result[0] is not None else "error: no output"


def outputs_match(actual, expected):
    # type: (str, str) -> bool
    return " ".join(actual.split()) == " ".join(expected.split())


def extract_code(text):
    # type: (str) -> Optional[str]
    matches = re.findall(r"```python(.*?)```", text, re.DOTALL)
    return matches[-1].strip() if matches else None


def _normalise(s):
    # type: (str) -> str
    s = s.replace("plaintext\n", "").replace("\\n", "\n")
    return s if s.endswith("\n") else s + "\n"


def extract_all_test_cases(text):
    # type: (str) -> List[Tuple[str, str]]
    """Extract all (input, output) pairs from a single model response.

    Tries backtick-fenced blocks first; falls back to plain-text patterns.
    Returns only pairs where both input and output are non-empty.
    """
    inps = re.findall(r'\*\*Test Input:\*\*\s*```(.*?)```', text, re.DOTALL)
    outs = re.findall(r'\*\*Test Output:\*\*\s*```(.*?)```', text, re.DOTALL)

    if inps and outs:
        pairs = [
            (_normalise(i.lstrip("\n")), _normalise(o.lstrip("\n")))
            for i, o in zip(inps, outs)
        ]
        return [(i, o) for i, o in pairs if i.strip() and o.strip()]

    # Plain-text fallback
    inps = re.findall(
        r'\*\*Test Input:\*\*\s*([\s\S]*?)(?=\*\*Test Output:\*\*)',
        text,
    )
    outs = re.findall(
        r'\*\*Test Output:\*\*\s*([\s\S]*?)(?=\*\*Explanation:|\*\*Test Input:|$)',
        text,
    )

    if inps and outs:
        pairs = [
            (_normalise(i.strip()), _normalise(o.strip()))
            for i, o in zip(inps, outs)
        ]
        return [(i, o) for i, o in pairs if i.strip() and o.strip()]

    return []


# ── Prompt builders ───────────────────────────────────────────────────────────

def _indent(s):
    # type: (str) -> str
    return "\n".join("    " + line for line in s.rstrip("\n").splitlines())


def _format_feedback(actuals):
    # type: (List[Tuple[str, str, str]]) -> str
    lines = []
    for i, (inp, actual, expected) in enumerate(actuals, 1):
        lines.append(f"Test {i}:")
        lines.append(f"  Input:\n{_indent(inp)}")
        lines.append(f"  Your output:     {actual.strip()}")
        lines.append(f"  Expected output: {expected.strip()}   <- may not be correct, use your judgment")
        lines.append("")
    return "\n".join(lines) + "\n"


def build_tester_prompt(problem, code, k_case):
    # type: (str, str, int) -> str
    return TESTER_TARGETED_TEMPLATE.format(problem=problem, code=code, k_case=k_case)


def build_refine_prompt(initial_prompt, prev_raw_responses, actuals):
    # type: (str, List[str], List[Tuple[str, str, str]]) -> str
    """Build a true multi-turn conversation prompt.

    Structure (history_turns=1):
        <initial_prompt><raw_response_0><|im_end|>
        <|im_start|>User: <feedback><|im_end|>
        <|im_start|>Assistant:

    To extend to history_turns>1, intermediate user-feedback turns would need
    to be stored and passed here as well. Currently only the latest turn is
    supported cleanly.
    """
    parts = [initial_prompt]
    for resp in prev_raw_responses:
        parts.append(resp)
        parts.append("<|im_end|>\n")
    feedback_msg = SOLVER_REFINE_FEEDBACK_TEMPLATE.format(
        feedback_block=_format_feedback(actuals)
    )
    parts.append(f"<|im_start|>User: {feedback_msg}<|im_end|>\n<|im_start|>Assistant: ")
    return "".join(parts)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _eval_on_gt(code, gt_inputs, gt_outputs, timeout, max_gt):
    # type: (Optional[str], List[str], List[str], float, int) -> Tuple[bool, List[bool]]
    if code is None:
        return False, []
    n = min(len(gt_inputs), max_gt)
    detail = [
        outputs_match(run_code(code, inp, timeout), exp)
        for inp, exp in zip(gt_inputs[:n], gt_outputs[:n])
    ]
    return (bool(detail) and all(detail)), detail


# ── Public API ────────────────────────────────────────────────────────────────

def _vprint(verbose, *args, **kwargs):
    # type: (bool, object, object) -> None
    if verbose:
        print(*args, **kwargs)


def _vblock(verbose, title, body):
    # type: (bool, str, str) -> None
    if not verbose:
        return
    sep = "=" * 72
    print(f"\n{sep}\n  {title}\n{sep}")
    print(body)
    print(sep)


def run_iterative_eval(
    sampler,
    tokenizer,
    problems,
    n_turns=2,
    k_case=3,
    max_new_tokens=1024,
    max_prompt_len=7168,
    temperature=0.8,
    top_p=0.95,
    exec_timeout=5.0,
    max_gt_test=5,
    history_turns=1,
    gt_solutions=None,
    verbose=False,
):
    """Run multi-turn iterative refinement eval.

    gt_solutions: optional dict mapping task_id (str or int) to a Python solution
      string.  When provided and a problem's ``task_id`` key matches an entry,
      the GT solution is executed on each tester-generated input and its output
      is compared against the tester's predicted output.  This measures tester
      accuracy without requiring the solver to be correct first.
      Raises RuntimeError if the GT solution errors or times out on any generated
      input — that indicates a bad data entry that should be fixed or removed.

    Set verbose=True to print every prompt sent to the model and every
    raw response received, plus the extracted code and test-execution results.

    Returns one result dict per problem (None if skipped — prompt too long):
        codes                 : list[str|None]        len = n_turns+1
        raw_responses         : list[str]             len = n_turns+1
        gen_tests             : list[list[tuple]]      len = n_turns
        actuals               : list[list[tuple]]      len = n_turns
        gt_pass_per_turn      : list[bool]            len = n_turns+1
        gt_detail_per_turn    : list[list[bool]]      len = n_turns+1
        gt_solution_used      : bool
        tester_vs_gt_per_turn : list[list[bool]]      len = n_turns
          Per generated test: True if tester predicted output matches GT output.
          Only populated when gt_solutions is provided and task_id is present.
    """
    def _sample(prompt, label):
        # type: (str, str) -> str
        _vblock(verbose, f"PROMPT  {label}", prompt)
        out = sampler(
            input_strings=[prompt],
            max_generation_steps=max_new_tokens,
            max_prompt_length=max_prompt_len,
            temperature=temperature,
            top_p=top_p,
        )
        resp = out.text[0]
        _vblock(verbose, f"RESPONSE {label}", resp)
        return resp

    results = []

    for prob_idx, prob in enumerate(problems):
        question    = prob["question"]
        gt_inputs   = prob["test_input"]
        gt_outputs  = prob["test_output"]
        time_limit  = float(prob.get("test_time_limit", 2.0))
        exec_to     = min(exec_timeout, time_limit)

        # Look up GT solution for this problem (task_id may be int or str in the dict).
        gt_sol = None
        if gt_solutions is not None:
            task_id = prob.get("task_id")
            if task_id is not None:
                gt_sol = gt_solutions.get(str(task_id)) or gt_solutions.get(task_id)

        prob_header = f"PROBLEM {prob_idx+1}/{len(problems)}"
        _vblock(verbose, prob_header, question[:500] + ("..." if len(question) > 500 else ""))

        initial_prompt = CODE_PROMPT_TEMPLATE.format(problem=question)
        if len(tokenizer.encode(initial_prompt)) > max_prompt_len:
            print(f"[{prob_idx+1:3d}/{len(problems)}] SKIP  initial prompt too long")
            results.append(None)
            continue

        # ── Turn 0: initial generation ────────────────────────────────────
        raw_0  = _sample(initial_prompt, f"prob={prob_idx+1} turn=0 SOLVER")
        code_0 = extract_code(raw_0)
        gt_pass_0, gt_detail_0 = _eval_on_gt(code_0, gt_inputs, gt_outputs, exec_to, max_gt_test)

        _vprint(verbose, f"\n  [turn 0] extracted code:\n{code_0}")
        _vprint(verbose, f"  [turn 0] gt_pass={gt_pass_0}  detail={gt_detail_0}")

        codes_hist           = [code_0]
        raw_resp_hist        = [raw_0]
        gen_tests_hist       = []
        actuals_hist         = []
        gt_pass_hist         = [gt_pass_0]
        gt_detail_hist       = [gt_detail_0]
        tester_vs_gt_hist    = []

        prev_code = code_0

        # ── Turns 1..n_turns ──────────────────────────────────────────────
        for _t in range(1, n_turns + 1):
            if prev_code is None:
                _vprint(verbose, f"\n  [turn {_t}] prev_code is None — skipping")
                codes_hist.append(None)
                raw_resp_hist.append("")
                gen_tests_hist.append([])
                actuals_hist.append([])
                gt_pass_hist.append(False)
                gt_detail_hist.append([])
                tester_vs_gt_hist.append([])
                continue

            # -- tester: one call, K tests in a single response ------------
            tester_prompt = build_tester_prompt(question, prev_code, k_case)
            if len(tokenizer.encode(tester_prompt)) > max_prompt_len:
                _vprint(verbose, f"\n  [turn {_t}] tester prompt too long — skipping tests")
                valid_tests = []
                raw_tester = ""
            else:
                raw_tester  = _sample(tester_prompt, f"prob={prob_idx+1} turn={_t} TESTER")
                valid_tests = extract_all_test_cases(raw_tester)
                _vprint(verbose, f"\n  [turn {_t}] tester extracted {len(valid_tests)} test(s)")

            gen_tests_hist.append(valid_tests)

            # -- execute current code on each generated test ---------------
            actuals = [
                (inp, run_code(prev_code, inp, exec_to), tester_exp)
                for inp, tester_exp in valid_tests
            ]
            actuals_hist.append(actuals)

            # -- evaluate tester accuracy against GT solution (metrics only) --
            tester_vs_gt = []
            if gt_sol is not None:
                for inp, tester_exp in valid_tests:
                    gt_out = run_code(gt_sol, inp, exec_to)
                    if gt_out == "timeout" or gt_out.startswith("error:"):
                        raise RuntimeError(
                            f"GT solution failed on generated input for prob={prob_idx+1} "
                            f"turn={_t}: {gt_out!r}\ninput={inp!r}\n"
                            "Fix or remove this data entry."
                        )
                    tester_vs_gt.append(outputs_match(tester_exp, gt_out))
            tester_vs_gt_hist.append(tester_vs_gt)

            if verbose:
                for i, (inp, actual, exp) in enumerate(actuals, 1):
                    match = outputs_match(actual, exp)
                    gt_label = f"  gt_match={tester_vs_gt[i-1]}" if tester_vs_gt else ""
                    print(f"  [turn {_t}] test {i}: {'PASS' if match else 'FAIL'}"
                          f"  actual={repr(actual.strip()[:60])}  expected={repr(exp.strip()[:60])}{gt_label}")

            # -- build multi-turn refine prompt ----------------------------
            context_responses = raw_resp_hist[-history_turns:]
            if actuals:
                refine_prompt = build_refine_prompt(initial_prompt, context_responses, actuals)
                # If too long, drop test cases from the end one by one
                while (
                    len(tokenizer.encode(refine_prompt)) > max_prompt_len
                    and actuals
                ):
                    actuals = actuals[:-1]
                    refine_prompt = (
                        build_refine_prompt(initial_prompt, context_responses, actuals)
                        if actuals else initial_prompt
                    )
            else:
                refine_prompt = initial_prompt

            # -- generate refined code -------------------------------------
            raw_t  = _sample(refine_prompt, f"prob={prob_idx+1} turn={_t} SOLVER")
            code_t = extract_code(raw_t)

            codes_hist.append(code_t)
            raw_resp_hist.append(raw_t)

            gt_pass_t, gt_detail_t = _eval_on_gt(code_t, gt_inputs, gt_outputs, exec_to, max_gt_test)
            gt_pass_hist.append(gt_pass_t)
            gt_detail_hist.append(gt_detail_t)

            _vprint(verbose, f"  [turn {_t}] extracted code:\n{code_t}")
            _vprint(verbose, f"  [turn {_t}] gt_pass={gt_pass_t}  detail={gt_detail_t}")

            prev_code = code_t

        results.append({
            "codes":                  codes_hist,
            "raw_responses":          raw_resp_hist,
            "gen_tests":              gen_tests_hist,
            "actuals":                actuals_hist,
            "gt_pass_per_turn":       gt_pass_hist,
            "gt_detail_per_turn":     gt_detail_hist,
            "gt_solution_used":       gt_sol is not None,
            "tester_vs_gt_per_turn":  tester_vs_gt_hist,
        })

        traj = "  ".join(
            ("PASS" if p else "FAIL") + f"(t{i})"
            for i, p in enumerate(gt_pass_hist)
        )
        print(f"[{prob_idx+1:3d}/{len(problems)}] {traj}")

    return results
