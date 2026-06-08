import os, subprocess, tempfile
from pathlib import Path
from dotenv import load_dotenv
import openai

load_dotenv(Path(__file__).parent.parent / ".env")
client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

resp = client.chat.completions.create(
    model="gpt-5.4-mini-2026-03-17",
    messages=[{"role": "user", "content": (
        "Solve the following competitive programming problem in Python 3.\n"
        "Output ONLY the Python 3 code, no explanation, no markdown fences.\n\n"
        "Problem:\nRead two integers A and B from a single line of stdin and print their sum.\n\n"
        "Example 1 Input:\n3 5\nExample 1 Output:\n8"
    )}],
    temperature=0,
)

raw = resp.choices[0].message.content
print("=== Raw response (repr) ===")
print(repr(raw))
print("\n=== Displayed ===")
print(raw)

def strip_fences(code):
    code = code.strip()
    if code.startswith("```"):
        lines = code.splitlines()
        code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return code

extracted = strip_fences(raw)
print("\n=== Extracted code ===")
print(extracted)

with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
    f.write(extracted)
    fname = f.name

r = subprocess.run(["python3", fname], input="3 5\n", capture_output=True, text=True)
os.unlink(fname)
print(f"\n=== Execution ===")
print(f"stdout: {repr(r.stdout)}")
print(f"stderr: {repr(r.stderr)}")
print(f"correct: {r.stdout.strip() == '8'}")
