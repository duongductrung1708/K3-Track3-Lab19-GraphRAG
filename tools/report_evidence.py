"""Thu 2 số liệu thật còn thiếu để viết lab_report.md §1 và §2.

§1 Coref: tái tạo CHÍNH XÁC extraction_source (400 chunk) của run 1 — deterministic,
   không cần LLM — rồi chạy coref THẬT trên các chunk có đại từ / generic reference
   để lấy chunk_id + unresolved_mentions có thật. Dùng model rẻ (pool fallback) vì
   openai/gpt-oss-120b đã cạn TPD trong run 1.

§2 Lexical Guard: lấy entity thật từ Neo4j (graph của run 1), embed bằng đúng model
   all-MiniLM-L6-v2, quét cặp similarity >= 0.85 rồi áp đúng merge_guard() của
   notebook. Hoàn toàn local, không tốn token.

Chạy: .venv/Scripts/python.exe tools/report_evidence.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
LAB_MAX_ARTICLES, LAB_MAX_CHUNKS, EXTRACTION_MAX_CHUNKS = 1500, 3000, 400
CHUNK_WORDS, CHUNK_OVERLAP_WORDS = 220, 40


def load_env():
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def norm_space(x):
    return re.sub(r"\s+", " ", str(x or "")).strip()


def sha1(x):
    return hashlib.sha1(str(x).encode("utf-8", errors="ignore")).hexdigest()


# ---------- tái tạo pipeline chunk của notebook (cell 008 / 012) ----------
def build_extraction_source():
    raw = pd.read_csv(ROOT / "data" / "hackernoon_subset.csv")
    raw["_source_row"] = np.arange(len(raw))

    det = pd.read_csv(ROOT / "data" / "graphrag_golden_50_first5000_detailed.csv")
    ev_rows = set()
    for s in det.get("evidence_row_ids_0based", pd.Series(dtype=str)).dropna():
        try:
            ev_rows.update(int(x) for x in json.loads(s))
        except Exception:
            pass

    df = pd.DataFrame()
    df["text"] = raw["description"].fillna("").map(norm_space)
    df["title"] = raw["title"].fillna("").map(norm_space)
    df["published_date"] = (
        pd.to_datetime(raw["published_at"], errors="coerce", utc=True)
        .dt.strftime("%Y-%m-%d").fillna("")
    )
    df["article_id"] = [sha1(f"{t}\n{x}")[:20] for t, x in zip(df["title"], df["text"])]
    df["source_row"] = raw["_source_row"].to_numpy()
    df = df[df["text"].str.len() >= 80].copy()
    df["dedup_key"] = [sha1(norm_space(f"{t}\n{x}").lower())
                       for t, x in zip(df["title"], df["text"])]
    before = len(df)
    df = df.drop_duplicates("dedup_key").drop(columns="dedup_key").reset_index(drop=True)

    must = df["source_row"].isin(ev_rows)
    keep = df[must]
    fill = df[~must].sample(frac=1.0, random_state=SEED).head(max(0, LAB_MAX_ARTICLES - len(keep)))
    news = pd.concat([keep, fill]).sort_values("source_row").reset_index(drop=True)
    print(f"[repro] dedup {before:,}->{len(df):,} | news_df={len(news):,} "
          f"({len(keep)} evidence + {len(fill)} distractor)")

    rows = []
    for r in news.itertuples(index=False):
        words = norm_space(r.text).split()
        step = max(1, CHUNK_WORDS - CHUNK_OVERLAP_WORDS)
        for i, start in enumerate(range(0, len(words), step)):
            part = words[start:start + CHUNK_WORDS]
            if not part:
                break
            rows.append({"chunk_id": f"{r.article_id}::c{i:04d}", "article_id": r.article_id,
                         "source_row": int(r.source_row), "title": r.title,
                         "published_date": r.published_date, "text": " ".join(part)})
            if start + CHUNK_WORDS >= len(words):
                break
        if len(rows) >= LAB_MAX_CHUNKS:
            break
    chunks = pd.DataFrame(rows)

    ev_articles = set(news.loc[news.source_row.isin(ev_rows), "article_id"])
    is_ev = chunks.article_id.isin(ev_articles)
    prio = chunks[is_ev]
    fillc = chunks[~is_ev].head(max(0, EXTRACTION_MAX_CHUNKS - len(prio)))
    sel = pd.concat([prio, fillc]).head(EXTRACTION_MAX_CHUNKS).reset_index(drop=True)
    ev_covered = sel.loc[sel.article_id.isin(ev_articles), "article_id"].nunique()
    print(f"[repro] chunks_df={len(chunks):,} | extraction_source={len(sel)} "
          f"| evidence article {ev_covered}/{len(ev_articles)}")
    return sel, ev_articles


# ---------- §1: coref thật trên chunk có đại từ ----------
COREF_SYSTEM = (
    "You are a conservative coreference-resolution component for a knowledge-graph pipeline.\n"
    "Resolve pronouns and generic references only when the antecedent is clearly supported in the same chunk.\n"
    "Never invent facts. Preserve dates, numbers, tickers and product names.\n"
    "Return strict JSON only."
)
PRONOUN = re.compile(
    r"\b(it|its|they|their|them|he|his|she|her|the company|the firm|"
    r"the startup|this deal|the deal|the acquisition|the platform)\b", re.I)


def probe_coref(sel, ev_articles, n=15, model=None, batch_size=5):
    cand = sel[sel.text.str.contains(PRONOUN, regex=True, na=False)]
    ev_first = cand[cand.article_id.isin(ev_articles)]
    pick = pd.concat([ev_first, cand[~cand.article_id.isin(ev_articles)]]).head(n)
    print(f"[coref] chunk có đại từ: {len(cand)}/{len(sel)} -> probe {len(pick)} chunk "
          f"(batch_size={batch_size}, giống notebook cell 012)")

    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    by_id, tok_in, tok_out = {}, 0, 0
    for start in range(0, len(pick), batch_size):
        batch = pick.iloc[start:start + batch_size]
        payload = [{"chunk_id": r.chunk_id, "text": r.text} for r in batch.itertuples(index=False)]
        prompt = ('Resolve coreferences.\n\nReturn:\n{\n  "items": [\n    {\n      "chunk_id": "...",\n'
                  '      "resolved_text": "...",\n      "unresolved_mentions": ["..."]\n    }\n  ]\n}\n\n'
                  f'INPUT:\n{json.dumps(payload, ensure_ascii=False)}')
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0.0, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": COREF_SYSTEM},
                          {"role": "user", "content": prompt}])
        except Exception as e:
            print(f"[coref] batch {start//batch_size} lỗi: {type(e).__name__}: {str(e)[:120]}")
            continue
        obj = json.loads(resp.choices[0].message.content)
        for x in obj.get("items", []):
            by_id[x.get("chunk_id")] = x
        tok_in += resp.usage.prompt_tokens
        tok_out += resp.usage.completion_tokens
    print(f"[coref] model={model} tokens prompt={tok_in} completion={tok_out} "
          f"| chunk có kết quả={len(by_id)}/{len(pick)}")

    out = []
    for r in pick.itertuples(index=False):
        it = by_id.get(r.chunk_id, {})
        res = norm_space(it.get("resolved_text") or r.text)
        unres = it.get("unresolved_mentions", []) or []
        out.append({"chunk_id": r.chunk_id, "source_row": r.source_row, "title": r.title,
                    "is_evidence": r.article_id in ev_articles,
                    "changed": res != norm_space(r.text), "n_unresolved": len(unres),
                    "unresolved": "; ".join(map(str, unres))[:300],
                    "orig": norm_space(r.text), "resolved": res})
    rep = pd.DataFrame(out)
    rep.to_csv(ROOT / "outputs" / "report_coref_probe.csv", index=False)
    print(f"[coref] changed={int(rep.changed.sum())}/{len(rep)} | "
          f"có unresolved={int((rep.n_unresolved > 0).sum())}/{len(rep)}")

    import difflib
    for r in rep.itertuples(index=False):
        if not r.changed and r.n_unresolved == 0:
            continue
        print(f"\n### {r.chunk_id} (row {r.source_row}, evidence={r.is_evidence})")
        print(f"    title: {r.title[:110]}")
        print(f"    unresolved_mentions: {r.unresolved or '(none)'}")
        ow, rw = r.orig.split(), r.resolved.split()
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ow, rw).get_opcodes():
            if tag == "equal":
                continue
            print(f"    [{tag}] GOC: ...{' '.join(ow[max(0,i1-10):i2+10])}...")
            print(f"          SAU: ...{' '.join(rw[max(0,j1-10):j2+10])}...")
    return rep


# ---------- §2: guard thật trên entity của graph ----------
CORP_SUFFIXES = {"inc", "incorporated", "corp", "corporation", "ltd", "limited",
                 "llc", "plc", "co", "company"}


def norm_entity(name):
    s = unicodedata.normalize("NFKC", norm_space(name)).lower()
    s = re.sub(r"[^\w\s\-\.]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def strip_suffix(name):
    toks = norm_entity(name).replace(".", "").split()
    while toks and toks[-1] in CORP_SUFFIXES:
        toks.pop()
    return " ".join(toks)


def merge_guard(a, b):
    na, nb = strip_suffix(a), strip_suffix(b)
    if na == nb:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= 0.72


def guard_scan(floor=0.85, top_k=5):
    from neo4j import GraphDatabase
    drv = GraphDatabase.driver(os.environ["NEO4J_URI"],
                               auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))
    with drv.session(database=os.environ.get("NEO4J_DATABASE", "neo4j")) as s:
        nodes = pd.DataFrame([r.data() for r in s.run(
            "MATCH (n:Entity) OPTIONAL MATCH (n)-[r]-() "
            "RETURN n.name AS name, n.entity_type AS type, count(r) AS degree")])
        counts = pd.DataFrame([r.data() for r in s.run(
            "MATCH (n:Entity) WITH count(n) AS nodes "
            "MATCH ()-[r]->() RETURN nodes, count(r) AS edges")])
        rel = pd.DataFrame([r.data() for r in s.run(
            "MATCH ()-[r]->() RETURN type(r) AS relation, count(*) AS n ORDER BY n DESC")])
        typ = pd.DataFrame([r.data() for r in s.run(
            "MATCH (n:Entity) RETURN n.entity_type AS type, count(*) AS n ORDER BY n DESC")])
        prov = pd.DataFrame([r.data() for r in s.run(
            "MATCH ()-[r]->() WHERE r.source_chunk_id IS NULL OR r.published_date IS NULL "
            "OR r.evidence IS NULL RETURN count(r) AS invalid")])
    drv.close()

    print("\n[graph]", counts.to_dict("records"), "| invalid_provenance =",
          prov.invalid.iloc[0])
    print("[graph] node theo type:", typ.to_dict("records"))
    print("[graph] relation:", rel.to_dict("records"))
    print("[graph] top degree:\n" + nodes.sort_values("degree", ascending=False)
          .head(8).to_string(index=False))

    from sentence_transformers import SentenceTransformer
    import faiss
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    rows = []
    for t, g in nodes.groupby("type"):
        names = g.name.tolist()
        if len(names) < 2:
            continue
        v = emb.encode(names, normalize_embeddings=True, show_progress_bar=False).astype("float32")
        idx = faiss.IndexFlatIP(v.shape[1])
        idx.add(v)
        sims, nbrs = idx.search(v, min(top_k, len(names)))
        for i in range(len(names)):
            for sc, j in zip(sims[i], nbrs[i]):
                if j < 0 or i >= j or float(sc) < floor:
                    continue
                ok = merge_guard(names[i], names[j])
                rows.append({"type": t, "left": names[i], "right": names[j],
                             "similarity": float(sc),
                             "lex_ratio": SequenceMatcher(
                                 None, strip_suffix(names[i]), strip_suffix(names[j])).ratio(),
                             "decision": "MERGE_VECTOR" if ok else "REJECT_GUARD",
                             "above_threshold_0.90": float(sc) >= 0.90})
    audit = pd.DataFrame(rows)
    if len(audit):
        audit = audit.sort_values("similarity", ascending=False)
        audit.to_csv(ROOT / "outputs" / "report_guard_scan.csv", index=False)
    print(f"\n[guard] cặp similarity >= {floor}: {len(audit)}")
    if len(audit):
        print(audit.to_string(index=False))
    return audit


def main():
    load_env()
    model = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-oss-20b"
    sel, ev = build_extraction_source()
    guard_scan()
    probe_coref(sel, ev, model=model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
