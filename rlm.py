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
    builtins = {
        "len": len,
        "range": range,
        "min": min,
        "max": max,
        "sum": sum,
        "sorted": sorted,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "any": any,
        "all": all,
        "set": set,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "print": print,
        "abs": abs,
        "round": round,
        "isinstance": isinstance,
        "type": type,
        "hasattr": hasattr,
        "getattr": getattr,
        "slice": slice,
    }
    # Add a fake import that just returns allowed modules
    allowed_modules = {"re": re}
    def safe_import(name, *args, **kwargs):
        if name in allowed_modules:
            return allowed_modules[name]
        raise ImportError(f"Import of '{name}' is not allowed")
    builtins["__import__"] = safe_import
    return builtins


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
        # Allow 're' import since we provide it
        ALLOWED_IMPORTS = {"re"}

        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in ALLOWED_IMPORTS:
                        raise SandboxError(f"Import of '{alias.name}' is disabled.")
            if isinstance(node, ast.ImportFrom):
                if node.module not in ALLOWED_IMPORTS:
                    raise SandboxError(f"Import from '{node.module}' is disabled.")
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


FINAL_RE = re.compile(r"FINAL\(([^)]*)\)", re.DOTALL)
FINAL_VAR_RE = re.compile(r"FINAL_VAR\(\s*([A-Za-z_]\w*)\s*\)", re.DOTALL)
# Match ```repl, ```python, or bare ``` code blocks
CODE_FENCE_RE = re.compile(r"```(?:repl|python)?\s*\n(.*?)\n```", re.DOTALL)


def _extract_code(model_output: str) -> tuple[str | None, bool]:
    """
    Extract code from model output. Returns (code, found_final).

    The paper uses ```repl blocks for code. We also accept ```python and bare ```.
    If FINAL() or FINAL_VAR() is found, returns (None, True) to signal termination.
    """
    model_output = model_output.strip()

    # Check for FINAL statements first
    if FINAL_RE.search(model_output) or FINAL_VAR_RE.search(model_output):
        return None, True

    # Look for code blocks
    matches = CODE_FENCE_RE.findall(model_output)
    if matches:
        # Join all code blocks if multiple
        return "\n\n".join(m.strip() for m in matches), False

    # If output looks like Python code (starts with common patterns), use it directly
    first_line = model_output.split('\n')[0].strip()
    code_indicators = ['import ', 'from ', 'def ', 'class ', 'for ', 'while ', 'if ',
                       'print(', 'context', 'llm_query', '#', 'result', 'answer', 'chunk']
    if any(first_line.startswith(ind) or first_line.startswith(ind.upper()) for ind in code_indicators):
        return model_output, False

    # Otherwise, not valid code - return None to prompt retry
    return None, False


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
        # Normalize common unicode characters that cause search issues
        # (PDFs often have en-dashes, smart quotes, etc.)
        normalized_prompt = prompt.replace('\u2013', '-').replace('\u2014', '-')  # en-dash, em-dash
        normalized_prompt = normalized_prompt.replace('\u2018', "'").replace('\u2019', "'")  # smart quotes
        normalized_prompt = normalized_prompt.replace('\u201c', '"').replace('\u201d', '"')

        # Initialize REPL environment with context variable (paper uses 'context')
        sandbox = PythonSandbox(initial_globals={"context": normalized_prompt})

        # Install helper functions that *the model can call from code*
        sandbox.globals.update(self._tooling(sandbox=sandbox, task=task, depth=0))

        system_prompt = self._system_prompt(prompt=normalized_prompt)
        user_prompt = self._user_prompt_intro(prompt=normalized_prompt, task=task)

        trace: list[RLMTraceStep] = []
        code_executed = False  # Track if model has explored the context

        for step in range(self.cfg.max_steps):
            model_out = self.root_lm.complete(system_prompt=system_prompt, user_prompt=user_prompt).strip()

            # Extract code or check for FINAL
            code, found_final = _extract_code(model_out)

            if found_final and not code_executed:
                # Don't allow FINAL before exploring the context
                obs = ""
                err = "You must explore the context with ```repl code BEFORE providing a FINAL answer. Write code to search/examine the context first."
                trace.append(RLMTraceStep(model_output=model_out, executed_code=None, observation=obs, error=err))
                user_prompt = self._next_user_prompt(task, step, obs, err)
                continue

            if found_final and code_executed:
                # Extract the actual answer from FINAL() or FINAL_VAR()
                m = FINAL_RE.search(model_out)
                if m:
                    ans = m.group(1).strip()
                    return RLMResult(answer=ans, trace=trace)

                m2 = FINAL_VAR_RE.search(model_out)
                if m2:
                    var = m2.group(1)
                    if var not in sandbox.globals:
                        obs = f"ERROR: variable {var!r} not found in environment."
                        trace.append(RLMTraceStep(model_output=model_out, executed_code=None, observation=obs, error=obs))
                        user_prompt = self._next_user_prompt(task, step, obs)
                        continue
                    ans = str(sandbox.globals[var])
                    return RLMResult(answer=ans, trace=trace)

            if code is None:
                # Model didn't output valid code - prompt to retry
                obs = ""
                err = "Please output Python code in ```repl blocks or provide FINAL(answer)/FINAL_VAR(varname)."
                trace.append(RLMTraceStep(model_output=model_out, executed_code=None, observation=obs, error=err))
                user_prompt = self._next_user_prompt(task, step, obs, err)
                continue

            # Execute the code
            exec_res = sandbox.exec(code)
            obs = exec_res.stdout.strip()
            err = exec_res.error

            # Mark that code has been executed (even if it had errors)
            if not err:
                code_executed = True

            # Truncate very long output
            if len(obs) > 8000:
                obs = obs[:8000] + "\n... [output truncated]"

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

        def llm_query(prompt: str) -> str:
            """
            Query a sub-LLM with the given prompt. The sub-LLM can handle ~500K chars.
            This is the paper's main recursive mechanism.
            """
            if depth >= cfg.max_recursion_depth:
                return "[llm_query blocked: max recursion depth reached]"

            prompt = prompt[: cfg.max_snippet_chars_for_subcall]

            key = (prompt, task)
            if cfg.cache_subcalls and key in self._subcall_cache:
                return self._subcall_cache[key]

            sys = (
                "You are a helpful sub-model. Answer the question or complete the task "
                "using ONLY the information provided in the prompt. Be concise and direct."
            )
            resp = self.sub_lm.complete(system_prompt=sys, user_prompt=prompt).strip()

            if cfg.cache_subcalls:
                self._subcall_cache[key] = resp
            return resp

        # Also keep 'subcall' as an alias for backwards compatibility
        def subcall(snippet: str, question: Optional[str] = None) -> str:
            q = question or task
            return llm_query(f"{q}\n\nContext:\n{snippet}")

        return {
            "llm_query": llm_query,
            "subcall": subcall,  # backwards compat
            "re": re,  # regex module for filtering
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "print": print,
        }

    # -----------------------------
    # Prompts
    # -----------------------------

    @staticmethod
    def _system_prompt(prompt: str) -> str:
        context_len = len(prompt)
        return f"""You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs.

Your context is a string with {context_len} total characters.

The REPL environment is initialized with:
1. A 'context' variable containing the full input text.
2. A 'llm_query(prompt)' function to query a sub-LLM (can handle ~500K chars).
3. The 'print()' function to view outputs.
4. The 're' module for regex operations.

When you want to execute Python code, wrap it in triple backticks with 'repl':
```repl
# Example: peek at first 500 chars
print(context[:500])
```

Example strategy for long contexts:
```repl
# Split into chunks and query each
chunk_size = len(context) // 5
answers = []
for i in range(5):
    start = i * chunk_size
    end = start + chunk_size if i < 4 else len(context)
    chunk = context[start:end]
    answer = llm_query(f"Find relevant info for the query in this chunk:\\n{{chunk}}")
    answers.append(answer)
    print(f"Chunk {{i}}: {{answer[:200]}}")

final_answer = llm_query(f"Combine these findings to answer the query:\\n" + "\\n".join(answers))
print(final_answer)
```

IMPORTANT RULES:
1. You MUST explore the context with ```repl code BEFORE answering
2. Do NOT use FINAL_VAR(x) unless you have already created variable x in a previous code execution
3. When you have found the answer, use FINAL(your answer here) directly - don't reference variables

Example workflow:
Step 1: ```repl code to search/explore
Step 2: See output, refine search if needed
Step 3: FINAL(the answer based on what you found)

Do NOT just say "I will do this" - execute your plan immediately in ```repl blocks."""

    @staticmethod
    def _user_prompt_intro(prompt: str, task: str) -> str:
        # Paper approach: tell the model about the context but DON'T show it
        # Force the model to use the REPL to explore
        num_lines = prompt.count('\n') + 1
        return f"""QUERY: {task}

CONTEXT INFO: {len(prompt):,} characters, ~{num_lines:,} lines.

IMPORTANT: The context is NOT shown here. You MUST write ```repl code to explore the `context` variable.
Start by examining relevant portions of the context to find information needed to answer the query."""

    @staticmethod
    def _next_user_prompt(task: str, step: int, stdout: str, err: Optional[str] = None) -> str:
        msg = f"QUERY: {task}\n\nSTEP {step} OUTPUT:\n"
        if stdout:
            msg += f"{stdout}\n"
        if err:
            msg += f"\nERROR:\n{err}\n"
        msg += "\nContinue with ```repl code, or provide FINAL(answer) / FINAL_VAR(varname)."
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
        print(f"TRACE ({len(result.trace)} steps)")
        print("=" * 60)
        for i, step in enumerate(result.trace):
            print(f"\n--- Step {i} ---")
            if step.executed_code:
                print(f"Code:\n{step.executed_code}")
            if step.observation:
                obs = step.observation[:1000] + "..." if len(step.observation) > 1000 else step.observation
                print(f"Output:\n{obs}")
            if step.error:
                print(f"Error: {step.error}")
        print("=" * 60)

    print(f"\nANSWER: {result.answer}")


if __name__ == "__main__":
    main()
