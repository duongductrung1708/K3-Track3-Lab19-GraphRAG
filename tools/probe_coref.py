"""Probe coreference resolution THẬT trên văn bản HackerNoon thật (mirror).

Mục đích: lấy 1 ca lỗi coref CÓ THẬT, ĐO ĐƯỢC để viết §1 lab_report.md, trong lúc
dataset gốc còn gated. Dùng đúng COREF_SYSTEM + đúng schema JSON như notebook
(cell 012) để kết quả có giá trị đối chiếu.

Chọn chunk cố ý: các row mà mirror nối liền nhiều bài báo không liên quan trong
cùng 1 field `description` -> chunk cắt ngang biên bài báo -> đại từ dễ bị gán
sai tiền ngữ (false antecedent) -> sinh false edge trong KG.

Chạy: .venv/Scripts/python.exe tools/probe_coref.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def norm_space(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


# --- giống hệt notebook cell 012 -------------------------------------------
COREF_SYSTEM = """
You are a conservative coreference-resolution component for a knowledge-graph pipeline.
Resolve pronouns and generic references only when the antecedent is clearly supported in the same chunk.
Never invent facts. Preserve dates, numbers, tickers and product names.
Return strict JSON only.
""".strip()

CHUNK_WORDS, CHUNK_OVERLAP = 220, 40


def chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP):
    words = norm_space(text).split()
    out, start = [], 0
    while start < len(words):
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += size - overlap
    return out


def main() -> int:
    load_env()
    key = os.environ.get("GROQ_API_KEY")
    model = os.environ.get("GROQ_MODEL")
    if not key:
        print("[FAIL] thiếu GROQ_API_KEY trong .env")
        return 1
    print(f"model = {model}")

    from groq import Groq
    client = Groq(api_key=key)

    df = pd.read_parquet(ROOT / ".preflight" / "mirror.parquet")
    d = df.description.astype(str)

    # row bị nối nhiều bài báo: <id 25-32 ký tự><timestamp><url> dính liền nhau
    joint = re.compile(r"[a-z0-9]{25,32}\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}https?://")
    cand = [i for i in range(len(df)) if len(joint.findall(d.iloc[i])) >= 1]
    print(f"row bị nối nhiều bài báo: {len(cand)} -> {cand}")

    payload, meta = [], {}
    for row in cand[:6]:
        for j, ck in enumerate(chunk_text(d.iloc[row])):
            cid = f"mirror{row}_c{j}"
            payload.append({"chunk_id": cid, "text": ck})
            meta[cid] = {"row": row, "companyName": df.companyName.iloc[row]}

    print(f"gửi {len(payload)} chunk đi coref\n")

    prompt = f"""
Resolve coreferences.

Return:
{{
  "items": [
    {{
      "chunk_id": "...",
      "resolved_text": "...",
      "unresolved_mentions": ["..."]
    }}
  ]
}}

INPUT:
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": COREF_SYSTEM},
                  {"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    obj = json.loads(resp.choices[0].message.content)
    usage = resp.usage

    by_id = {x.get("chunk_id"): x for x in obj.get("items", [])}
    src = {p["chunk_id"]: p["text"] for p in payload}

    out_rows = []
    for cid, orig in src.items():
        item = by_id.get(cid, {})
        res = norm_space(item.get("resolved_text") or orig)
        unres = item.get("unresolved_mentions", [])
        changed = res != norm_space(orig)
        out_rows.append({
            "chunk_id": cid, "row": meta[cid]["row"],
            "companyName": meta[cid]["companyName"],
            "changed": changed, "n_unresolved": len(unres),
            "unresolved": "; ".join(map(str, unres))[:200],
            "orig": norm_space(orig), "resolved": res,
        })

    rep = pd.DataFrame(out_rows)
    rep.to_csv(ROOT / ".preflight" / "coref_probe.csv", index=False)

    print("=" * 100)
    print(f"chunk gửi={len(src)}  chunk LLM trả về={len(by_id)}  "
          f"bị đổi text={int(rep.changed.sum())}  "
          f"có unresolved_mentions={int((rep.n_unresolved > 0).sum())}")
    print(f"tokens: prompt={usage.prompt_tokens} completion={usage.completion_tokens}")
    print("=" * 100)

    for r in rep.itertuples(index=False):
        if not r.changed and r.n_unresolved == 0:
            continue
        print(f"\n### {r.chunk_id}  (row {r.row}, companyName={r.companyName!r})")
        print(f"  unresolved_mentions: {r.unresolved or '(none)'}")
        # in ra đúng chỗ khác nhau
        ow, rw = r.orig.split(), r.resolved.split()
        import difflib
        sm = difflib.SequenceMatcher(None, ow, rw)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            print(f"  [{tag:7s}] GỐC   : ...{' '.join(ow[max(0,i1-12):i2+12])}...")
            print(f"            SAU   : ...{' '.join(rw[max(0,j1-12):j2+12])}...")

    print(f"\n-> đã ghi .preflight/coref_probe.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
