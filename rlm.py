"""
Recursive Language Model (RLM) scaffold inspired by:
"Recursive Language Models" (Zhang, Kraska, Khattab, 2025)

Core idea:
- Load the full prompt as variable P in a Python "environment".
- The LM writes Python code to peek/search/chunk P and can recursively sub-call
  itself over snippets.
- The external interface remains: input string -> output string.

This file is intentionally provider-agnostic: plug in any LM API you want.
"""

from __future__ import annotations

import ast
import io
import re
import textwrap
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple


# -----------------------------
# LM interface (provider-agnostic)
# -----------------------------

class LMClient:
    """
    Minimal interface. Implement `complete(system_prompt, user_prompt) -> str`.
    The return is the model's proposed *Python code* OR FINAL(...) / FINAL_VAR(...).
    """
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


# -----------------------------
# Sandbox execution environment
# -----------------------------

class SandboxError(Exception):
    pass


def _safe_builtins() -> Dict[str, Any]:
    """
    Keep this small. The model can still do plenty with strings/regex/lists,
    but cannot import modules, open files, etc.
    """
    return {
        "len": len,
        "range": range,
        "min": min,
        "max": max,
        "sum": sum,
        "sorted": sorted,
        "enumerate": enumerate,
        "zip": zip,
        "set": set,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "print": print,
        # Regex is important for the paper’s “filtering via code” pattern
        "re": re,
    }


@dataclass
class ExecutionResult:
    stdout: str
    error: Optional[str] = None


class PythonSandbox:
    """
    A minimal REPL-like executor with:
    - persistent globals
    - restricted builtins
    - stdout capture
    """
    def __init__(self, initial_globals: Optional[Dict[str, Any]] = None):
        self.globals: Dict[str, Any] = {}
        self.globals["__builtins__"] = _safe_builtins()
        if initial_globals:
            self.globals.update(initial_globals)

    def exec(self, code: str) -> ExecutionResult:
        code = textwrap.dedent(code).strip()
        if not code:
            return ExecutionResult(stdout="", error=None)

        # Basic guardrails: block import / attribute access tricks in a simple way.
        # (For real deployments, use a proper sandbox.)
        self._basic_static_checks(code)

        buf = io.StringIO()
        try:
            compiled = compile(code, "<rlm>", "exec")
            # Capture stdout
            import contextlib
            with contextlib.redirect_stdout(buf):
                exec(compiled, self.globals, self.globals)
            return ExecutionResult(stdout=buf.getvalue(), error=None)
        except Exception as e:
            tb = traceback.format_exc(limit=3)
            return ExecutionResult(stdout=buf.getvalue(), error=f"{type(e).__name__}: {e}\n{tb}")

    @staticmethod
    def _basic_static_checks(code: str) -> None:
        """
        Very lightweight checks. Not bulletproof.
        """
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise SandboxError("Imports are disabled in this sandbox.")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"open", "exec", "eval", "__import__"}:
                    raise SandboxError(f"Call to {node.func.id} is disabled.")
            # Disallow dunder attribute access which can be abused
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise SandboxError("Dunder attribute access is disabled.")


# -----------------------------
# RLM core
# -----------------------------

@dataclass
class RLMConfig:
    max_steps: int = 12
    max_recursion_depth: int = 1
    max_snippet_chars_for_subcall: int = 20_000
    max_peek_chars: int = 2_000
    cache_subcalls: bool = True


@dataclass
class RLMTraceStep:
    model_output: str
    executed_code: Optional[str]
    observation: str
    error: Optional[str] = None


@dataclass
class RLMResult:
    answer: str
    trace: list[RLMTraceStep] = field(default_factory=list)


FINAL_RE = re.compile(r"^\s*FINAL\((.*)\)\s*$", re.DOTALL)
FINAL_VAR_RE = re.compile(r"^\s*FINAL_VAR\(\s*([A-Za-z_]\w*)\s*\)\s*$", re.DOTALL)
CODE_FENCE_RE = re.compile(r"^```(?:python)?\s*\n(.*?)\n```\s*$", re.DOTALL)


def _extract_code(model_output: str) -> str:
    """
    Strip markdown code fences if present - some models wrap code in ```python ... ```.
    """
    model_output = model_output.strip()
    m = CODE_FENCE_RE.match(model_output)
    if m:
        return m.group(1).strip()
    return model_output


class RecursiveLanguageModel:
    """
    Implements the paper’s "prompt as environment" mechanism:
    - P is loaded as a variable in the environment
    - LM writes code to inspect/decompose P and optionally subcall() on snippets
    """

    def __init__(self, root_lm: LMClient, sub_lm: Optional[LMClient] = None, config: Optional[RLMConfig] = None):
        self.root_lm = root_lm
        self.sub_lm = sub_lm or root_lm
        self.cfg = config or RLMConfig()
        self._subcall_cache: Dict[Tuple[str, str], str] = {}

    def run(self, prompt: str, task: str) -> RLMResult:
        # Initialize REPL environment with prompt P
        sandbox = PythonSandbox(initial_globals={"P": prompt})

        # Install helper functions that *the model can call from code*
        sandbox.globals.update(self._tooling(sandbox=sandbox, task=task, depth=0))

        system_prompt = self._system_prompt()
        user_prompt = self._user_prompt_intro(prompt=prompt, task=task)

        trace: list[RLMTraceStep] = []

        for step in range(self.cfg.max_steps):
            model_out = self.root_lm.complete(system_prompt=system_prompt, user_prompt=user_prompt).strip()

            # 1) If model returns FINAL(...)
            m = FINAL_RE.match(model_out)
            if m:
                ans = m.group(1).strip()
                return RLMResult(answer=ans, trace=trace)

            # 2) If model returns FINAL_VAR(name) (paper-style tag)
            m2 = FINAL_VAR_RE.match(model_out)
            if m2:
                var = m2.group(1)
                if var not in sandbox.globals:
                    obs = f"ERROR: variable {var!r} not found in environment."
                    trace.append(RLMTraceStep(model_output=model_out, executed_code=None, observation=obs, error=obs))
                    user_prompt = self._next_user_prompt(task, step, obs)
                    continue
                ans = str(sandbox.globals[var])
                return RLMResult(answer=ans, trace=trace)

            # 3) Otherwise treat output as Python code to execute
            code = _extract_code(model_out)
            exec_res = sandbox.exec(code)
            obs = exec_res.stdout.strip()
            err = exec_res.error

            trace.append(RLMTraceStep(
                model_output=model_out,
                executed_code=code,
                observation=obs,
                error=err
            ))

            # Feed observation back
            user_prompt = self._next_user_prompt(task, step, obs, err)

        return RLMResult(
            answer=f"[RLM stopped after {self.cfg.max_steps} steps without FINAL() / FINAL_VAR().]",
            trace=trace
        )

    # -----------------------------
    # Tooling exposed inside the REPL
    # -----------------------------

    def _tooling(self, sandbox: PythonSandbox, task: str, depth: int) -> Dict[str, Any]:
        cfg = self.cfg

        def peek(start: int = 0, length: int = 500) -> str:
            length = max(0, min(length, cfg.max_peek_chars))
            return sandbox.globals["P"][start:start + length]

        def lines(i: int = 0, j: int = 10) -> str:
            parts = sandbox.globals["P"].splitlines()
            i = max(0, i)
            j = min(len(parts), j)
            return "\n".join(parts[i:j])

        def grep(pattern: str, max_matches: int = 5, context: int = 80) -> str:
            # Simple regex search over P; returns small excerpts.
            text = sandbox.globals["P"]
            out = []
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                s = max(0, m.start() - context)
                e = min(len(text), m.end() + context)
                out.append(text[s:e].replace("\n", " "))
                if len(out) >= max_matches:
                    break
            return "\n---\n".join(out) if out else ""

        def chunk_by_chars(size: int = 4000, overlap: int = 200) -> list[str]:
            text = sandbox.globals["P"]
            size = max(200, size)
            overlap = max(0, min(overlap, size // 2))
            chunks = []
            i = 0
            while i < len(text):
                chunks.append(text[i:i + size])
                i += (size - overlap)
            return chunks

        def chunk_by_lines(lines_per_chunk: int = 50, overlap: int = 5) -> list[str]:
            """
            Split P into chunks by line count - useful for line-oriented tasks like OOLONG.
            """
            all_lines = sandbox.globals["P"].splitlines()
            lines_per_chunk = max(1, lines_per_chunk)
            overlap = max(0, min(overlap, lines_per_chunk // 2))
            chunks = []
            i = 0
            while i < len(all_lines):
                chunks.append("\n".join(all_lines[i:i + lines_per_chunk]))
                i += (lines_per_chunk - overlap)
            return chunks

        def subcall(snippet: str, question: Optional[str] = None) -> str:
            """
            Recursively call a sub-LM on a smaller snippet.
            Mirrors the paper's "recursive call over snippets" mechanism.
            """
            if depth >= cfg.max_recursion_depth:
                return "[subcall blocked: max recursion depth reached]"

            snippet = snippet[: cfg.max_snippet_chars_for_subcall]
            q = question or task

            key = (snippet, q)
            if cfg.cache_subcalls and key in self._subcall_cache:
                return self._subcall_cache[key]

            # In a deeper recursion setting, you could call another RecursiveLanguageModel here.
            # For "depth=1" (the paper’s common setup), treat subcalls as plain LM calls.
            sys = (
                "You are a sub-model. Answer the question using ONLY the provided snippet. "
                "If the snippet is insufficient, say what is missing."
            )
            user = f"QUESTION:\n{q}\n\nSNIPPET:\n{snippet}"
            resp = self.sub_lm.complete(system_prompt=sys, user_prompt=user).strip()

            if cfg.cache_subcalls:
                self._subcall_cache[key] = resp
            return resp

        return {
            "peek": peek,
            "lines": lines,
            "grep": grep,
            "chunk_by_chars": chunk_by_chars,
            "chunk_by_lines": chunk_by_lines,
            "subcall": subcall,
            "TASK": task,
            "DEPTH": depth,
        }

    # -----------------------------
    # Prompts
    # -----------------------------

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are operating as a Recursive Language Model inside a Python REPL.\n"
            "The full user input is stored as variable P (a string) in the environment.\n"
            "You can write Python code to inspect P using helper functions:\n"
            "- peek(start, length) - view substring of P\n"
            "- lines(i, j) - view lines i to j of P\n"
            "- grep(pattern, max_matches=5, context=80) - regex search with context\n"
            "- chunk_by_chars(size, overlap) - split P into overlapping char chunks\n"
            "- chunk_by_lines(lines_per_chunk, overlap) - split P into line-based chunks\n"
            "- subcall(snippet, question=...) - delegate to a sub-model on a snippet\n\n"
            "Rules:\n"
            "- Prefer selective inspection (grep/peek) over printing huge text.\n"
            "- Use subcall() for semantic tasks on chunks that need LM reasoning.\n"
            "- When ready to answer, output FINAL(<answer>) or FINAL_VAR(<varname>).\n"
            "- Output ONLY Python code OR a FINAL/FINAL_VAR statement. No explanations.\n"
        )

    @staticmethod
    def _user_prompt_intro(prompt: str, task: str) -> str:
        return (
            f"TASK:\n{task}\n\n"
            f"ENV INFO:\n- len(P) = {len(prompt)} characters\n"
            "Write Python code to solve the task by selectively inspecting P.\n"
        )

    @staticmethod
    def _next_user_prompt(task: str, step: int, stdout: str, err: Optional[str] = None) -> str:
        msg = f"TASK:\n{task}\n\nSTEP {step} OBSERVATION:\n"
        if stdout:
            msg += f"{stdout}\n"
        if err:
            msg += f"\nERROR:\n{err}\n"
        msg += "\nWrite code to continue, or output FINAL(<answer>) / FINAL_VAR(<varname>)."
        return msg


# -----------------------------
# LiteLLM-based LMClient
# -----------------------------

class LiteLLMClient(LMClient):
    """
    LMClient implementation using litellm for provider-agnostic LLM calls.

    Supports any litellm model string format:
    - "openrouter/anthropic/claude-sonnet-4" (requires OPENROUTER_API_KEY)
    - "anthropic/claude-sonnet-4-20250514" (requires ANTHROPIC_API_KEY)
    - "openai/gpt-4o" (requires OPENAI_API_KEY)
    """

    def __init__(self, model: str = "openrouter/anthropic/claude-sonnet-4", **kwargs):
        self.model = model
        self.kwargs = kwargs

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        import litellm

        response = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **self.kwargs
        )
        return response.choices[0].message.content or ""


# -----------------------------
# CLI / Demo
# -----------------------------

def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Recursive Language Model CLI")
    parser.add_argument("--model", default="openrouter/anthropic/claude-sonnet-4",
                        help="Root LM model (litellm format, e.g. openrouter/anthropic/claude-sonnet-4)")
    parser.add_argument("--sub-model", default=None,
                        help="Sub-LM model for recursive calls (defaults to root model)")
    parser.add_argument("--task", required=True, help="Task/question to answer")
    parser.add_argument("--input", "-i", help="Input file (or stdin if omitted)")
    parser.add_argument("--max-steps", type=int, default=12, help="Max REPL steps")
    parser.add_argument("--max-depth", type=int, default=1, help="Max recursion depth")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show trace")

    args = parser.parse_args()

    # Read input
    if args.input:
        with open(args.input, "r") as f:
            prompt = f.read()
    else:
        prompt = sys.stdin.read()

    # Initialize LM clients
    root_lm = LiteLLMClient(model=args.model)
    sub_lm = LiteLLMClient(model=args.sub_model) if args.sub_model else root_lm

    config = RLMConfig(
        max_steps=args.max_steps,
        max_recursion_depth=args.max_depth,
    )

    rlm = RecursiveLanguageModel(root_lm=root_lm, sub_lm=sub_lm, config=config)
    result = rlm.run(prompt=prompt, task=args.task)

    if args.verbose:
        print("=" * 60)
        print("TRACE")
        print("=" * 60)
        for i, step in enumerate(result.trace):
            print(f"\n--- Step {i} ---")
            print(f"Code:\n{step.executed_code}")
            print(f"Output:\n{step.observation}")
            if step.error:
                print(f"Error:\n{step.error}")
        print("=" * 60)

    print(f"\nANSWER: {result.answer}")


if __name__ == "__main__":
    main()
