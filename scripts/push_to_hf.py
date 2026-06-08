"""
Merge py3_solutions_test.jsonl with Gen-Verse/CodeContests and push to HuggingFace.
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi

load_dotenv(Path(__file__).parent.parent / ".env")
HF_TOKEN = os.environ["HF_TOKEN"]
REPO_ID = "sunnytqin/CodeContests_with_py3"
JSONL_PATH = Path(__file__).parent.parent / "py3_solutions_test.jsonl"

# Load the solutions we generated
solutions = {}
with open(JSONL_PATH) as f:
    for line in f:
        r = json.loads(line)
        solutions[r["task_id"]] = {"py3_solution": r["py3_solution"], "source": r["source"]}

print(f"Loaded {len(solutions)} solutions from jsonl")

# Load the original Gen-Verse test split
gv = load_dataset("Gen-Verse/CodeContests", split="test")
print(f"Loaded Gen-Verse/CodeContests test: {len(gv)} rows")

# Merge
records = []
for ex in gv:
    sol_info = solutions.get(ex["task_id"], {"py3_solution": None, "source": "missing"})
    records.append({**ex, **sol_info})

ds = Dataset.from_list(records)
print(f"Merged dataset: {len(ds)} rows, columns: {ds.column_names}")

# Push
api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)

ds.push_to_hub(REPO_ID, split="test", token=HF_TOKEN)
print(f"Pushed to https://huggingface.co/datasets/{REPO_ID}")
