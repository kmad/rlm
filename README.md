# Recursive Language Models (RLM)

Python implementation of **Recursive Language Models**, enabling LLMs to process arbitrarily long prompts by treating them as an external environment. The model writes Python code to examine, decompose, and recursively call itself over snippets.

Based on: [Recursive Language Models](https://arxiv.org/abs/2512.24601v1) (Zhang, Kraska, Khattab, 2025)

## Installation

```bash
uv sync
```

## Usage

```bash
uv run rlm.py \
  --model openrouter/anthropic/claude-sonnet-4.5 \
  --sub-model openrouter/grok-4-fast \
  --task "Given the input, what is the definition of a small bank under CRA? Respond with JUST the dollar amount." \
  -i data/fed_reg_all_5mtok.txt \
  --max-steps 10 \
  -v
```

**Arguments:**
- `--model`: Root LLM model (LiteLLM format)
- `--sub-model`: Sub-LM for recursive calls (optional, defaults to root model)
- `--task`: Task/question to answer
- `-i, --input`: Input file (or use stdin)
- `--max-steps`: Max REPL steps (default: 12)
- `--max-depth`: Max recursion depth (default: 1)
- `-v, --verbose`: Show execution trace

## How It Works

1. The full prompt is loaded as `context` in a Python sandbox
2. The LLM writes Python code (in ````repl` blocks) to explore `context`
3. Code executes in the sandbox; observations feed back to the model
4. For long contexts, the model can use `llm_query()` to recursively query sub-LLMs on chunks
5. Model signals completion with `FINAL(answer)` or `FINAL_VAR(varname)`

## Python API

```python
from rlm import RecursiveLanguageModel, LiteLLMClient, RLMConfig

root_lm = LiteLLMClient(model="openrouter/anthropic/claude-sonnet-4.5")
rlm = RecursiveLanguageModel(root_lm=root_lm, config=RLMConfig(max_steps=10))
result = rlm.run(prompt=prompt, task="Your question here")
print(result.answer)
```

## API Keys

Set environment variables for your provider:
```bash
export OPENROUTER_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
```

## Examples

See the `data/` directory for sample documents. Example queries:

```bash
# Question answering on long documents
python rlm.py --task "What regulations were announced?" --input data/FederalRegister-2026-01-05.txt

# Information extraction
python rlm.py --task "List all dates mentioned in the document" --input data/FR-2025-12-17.txt

# Summarization
python rlm.py --task "Provide a 3-paragraph summary" --input data/paper.txt --verbose
```

## Architecture

```
┌─────────────────┐
│   User Query    │
│   + Context     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RLM Scaffold   │
│  ┌───────────┐  │
│  │  Sandbox  │  │ ← context variable
│  │  + Tools  │  │ ← llm_query(), re, etc.
│  └─────┬─────┘  │
│        │        │
│        ▼        │
│  ┌───────────┐  │
│  │ Root LLM  │  │ ← Generates Python code
│  └─────┬─────┘  │
│        │        │
│        ▼        │
│  ┌───────────┐  │
│  │ Execute   │  │ ← Run code in sandbox
│  └─────┬─────┘  │
│        │        │
│        ▼        │
│  ┌───────────┐  │
│  │ Sub-LLM   │  │ ← Recursive calls on chunks
│  └───────────┘  │
└─────────────────┘
```

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{zhang2025recursive,
  title={Recursive Language Models},
  author={Zhang, Alex L. and Kraska, Tim and Khattab, Omar},
  journal={arXiv preprint arXiv:2512.24601},
  year={2025}
}
```
