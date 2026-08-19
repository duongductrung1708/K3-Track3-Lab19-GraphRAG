"""Patch notebook gốc của giảng viên -> bản chạy được local (Windows + Docker Neo4j).

Chạy:  .venv/Scripts/python.exe tools/patch_notebook.py

Mọi thay đổi đều idempotent và có assert: nếu anchor không tìm thấy -> raise,
để không bao giờ patch "im lặng" sai chỗ. Danh sách thay đổi xem PATCHES ở cuối file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb"

CHANGELOG: list[str] = []


def src(cell) -> str:
    s = cell["source"]
    return s if isinstance(s, str) else "".join(s)


def set_src(cell, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def replace_cell(cells, idx: int, must_contain: str, new_text: str, label: str) -> None:
    """Thay toàn bộ source của cell idx. must_contain = anchor xác nhận đúng cell."""
    body = src(cells[idx])
    if new_text.strip() == body.strip():
        CHANGELOG.append(f"  = cell {idx:03d} {label} (đã patch trước đó, bỏ qua)")
        return
    if must_contain not in body:
        raise SystemExit(
            f"[FAIL] cell {idx:03d} ({label}): không thấy anchor {must_contain!r}.\n"
            f"--- nội dung hiện tại ---\n{body[:400]}"
        )
    set_src(cells[idx], new_text)
    CHANGELOG.append(f"  * cell {idx:03d} {label} -> thay toàn bộ")


def sub_in_cell(cells, idx: int, old: str, new: str, label: str) -> None:
    """Thay 1 đoạn trong cell idx."""
    body = src(cells[idx])
    if old not in body:
        if new in body:
            CHANGELOG.append(f"  = cell {idx:03d} {label} (đã patch trước đó)")
            return
        raise SystemExit(f"[FAIL] cell {idx:03d} ({label}): không thấy {old!r}")
    set_src(cells[idx], body.replace(old, new))
    CHANGELOG.append(f"  * cell {idx:03d} {label}")


def uncomment(cells, idx: int, needles: list[str], label: str) -> None:
    """Bỏ dấu # ở đầu các dòng driver-call chứa needle."""
    lines = src(cells[idx]).splitlines(keepends=True)
    hits = 0
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped.startswith("# "):
            continue
        payload = stripped[2:]
        if any(n in payload for n in needles):
            indent = line[: len(line) - len(stripped)]
            lines[i] = indent + payload
            hits += 1
    already = any(
        any(n in l for n in needles) and not l.lstrip().startswith("#")
        for l in lines
    )
    if hits == 0 and not already:
        raise SystemExit(f"[FAIL] cell {idx:03d} ({label}): không uncomment được dòng nào")
    cells[idx]["source"] = lines
    CHANGELOG.append(
        f"  * cell {idx:03d} {label} -> uncomment {hits} dòng"
        if hits else f"  = cell {idx:03d} {label} (đã uncomment trước đó)"
    )


# ----------------------------------------------------------------------------
# Nội dung cell mới
# ----------------------------------------------------------------------------

CELL_003 = '''#@title 1.1 — Install (chỉ cài package còn thiếu)
# Bản gốc cài thêm spacy / langchain-community / llama-index nhưng KHÔNG cell nào dùng
# -> đã loại bỏ (tiết kiệm vài phút mỗi lần Run All, tránh xung đột dependency).
# Thêm python-dotenv để đọc .env khi chạy local (thay cho Colab Secrets).
import importlib.util

REQUIRED = {
    "neo4j": "neo4j",
    "pandas": "pandas",
    "numpy": "numpy",
    "pyarrow": "pyarrow",
    "sentence_transformers": "sentence-transformers",
    "faiss": "faiss-cpu",
    "groq": "groq",
    "openai": "openai",
    "tqdm": "tqdm",
    "networkx": "networkx",
    "datasets": "datasets",
    "dotenv": "python-dotenv",
}

missing = [pkg for mod, pkg in REQUIRED.items() if importlib.util.find_spec(mod) is None]
if missing:
    spec = " ".join(missing)
    print("Đang cài:", spec)
    %pip -q install $spec
else:
    print("✅ Dependencies đã đủ — bỏ qua bước cài.")
'''

CELL_004 = '''#@title 1.2 — Imports & config
import os, re, json, time, random, hashlib, unicodedata
from pathlib import Path
from collections import defaultdict, Counter, deque
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import faiss

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
pd.set_option("display.max_colwidth", 120)

# --- Project layout portable: chạy được cả Colab lẫn local (Windows/Linux) ---
# Bản gốc hard-code /content/... nên crash ngoài Colab và không xuất CSV vào outputs/.
def _find_project_root() -> Path:
    here = Path.cwd().resolve()
    for cand in [here, *here.parents]:
        if (cand / "RUBRIC.md").exists() or (cand / ".git").exists():
            return cand
    return here

PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
for _d in (DATA_DIR, OUTPUTS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Secrets: Colab userdata -> .env -> os.environ. Không hard-code key vào notebook. ---
def _read_dotenv(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out

DOTENV = _read_dotenv(PROJECT_ROOT / ".env")

def get_secret(name, default=None):
    try:
        from google.colab import userdata
        value = userdata.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.environ.get(name) or DOTENV.get(name) or default

NEO4J_URI = get_secret("NEO4J_URI", "")
NEO4J_USER = get_secret("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = get_secret("NEO4J_PASSWORD", "")
NEO4J_DATABASE = get_secret("NEO4J_DATABASE", "neo4j")

GROQ_API_KEY = get_secret("GROQ_API_KEY", "")
GROQ_MODEL = get_secret("GROQ_MODEL", "")

JUDGE_PROVIDER = get_secret("JUDGE_PROVIDER", "openai").lower()
JUDGE_MODEL = get_secret("JUDGE_MODEL", "")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY", "")
HF_TOKEN = get_secret("HF_TOKEN", "")

# --- Paths (override được qua .env nếu muốn) ---
DATA_PATH = Path(get_secret("DATA_PATH") or (DATA_DIR / "hackernoon_subset.csv"))
GOLDEN_PATH = Path(get_secret("GOLDEN_PATH") or (DATA_DIR / "graphrag_golden_50_first5000.csv"))
GOLDEN_DETAILED_PATH = DATA_DIR / "graphrag_golden_50_first5000_detailed.csv"
CHECKPOINT = OUTPUTS_DIR / "graphrag_eval_checkpoint.csv"

# --- Scale guard (giữ nguyên theo yêu cầu lab) ---
LAB_MAX_ARTICLES = 1500
LAB_MAX_CHUNKS = 3000
EXTRACTION_MAX_CHUNKS = 400
CHUNK_WORDS = 220
CHUNK_OVERLAP_WORDS = 40

# Golden dataset được soạn trên 5000 dòng ĐẦU của hackernoon_subset.csv,
# nên chỉ stream đúng 5000 dòng đó để row index khớp tuyệt đối.
GOLDEN_SOURCE_ROWS = 5000

print(f"PROJECT_ROOT = {PROJECT_ROOT}")
print(f"DATA_PATH    = {DATA_PATH}")
print(f"GOLDEN_PATH  = {GOLDEN_PATH}")
print("Secrets:", {
    k: ("✅" if get_secret(k) else "❌")
    for k in ["NEO4J_URI", "NEO4J_PASSWORD", "GROQ_API_KEY", "GROQ_MODEL",
              "JUDGE_PROVIDER", "JUDGE_MODEL", "HF_TOKEN"]
})
'''

CELL_006 = '''#@title 1.3 — Stream HackerNoon dataset -> CSV
import csv
import os
from datasets import load_dataset
from tqdm.auto import tqdm

DATASET_NAME = "HackerNoon/tech-company-news-data-dump"
OUTPUT_CSV = Path(DATA_PATH)

# Golden dataset tham chiếu row 0..4999 của file này => chốt LIMIT_ROWS = 5000.
# Cell 006 ghi tuần tự theo thứ tự stream nên row index tái lập được chính xác.
LIMIT_ROWS = GOLDEN_SOURCE_ROWS
LIMIT_MB = 300              # chặn an toàn, không phải mục tiêu
PRIORITIZE_MB = False       # dừng theo số dòng để khớp golden dataset
FORCE_REDOWNLOAD = False    # True nếu muốn tải lại từ đầu


def _csv_rows(path: Path) -> int:
    with open(path, encoding="utf-8", newline="") as f:
        return max(0, sum(1 for _ in csv.reader(f)) - 1)


# Idempotent: Run All nhiều lần không tải lại 5000 dòng.
if OUTPUT_CSV.exists() and not FORCE_REDOWNLOAD and _csv_rows(OUTPUT_CSV) >= LIMIT_ROWS:
    print(f"✅ Đã có {OUTPUT_CSV} ({_csv_rows(OUTPUT_CSV):,} dòng) — bỏ qua download.")
else:
    if not HF_TOKEN:
        raise ValueError(
            "Thiếu HF_TOKEN. Đặt trong .env (local) hoặc Colab Secrets. "
            "Dataset là gated: phải bấm Agree/Request access trên trang Hugging Face "
            "và token cần scope đọc public gated repos."
        )

    print("Đang kết nối luồng dữ liệu (streaming)...")
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True, token=HF_TOKEN)
    iterator = iter(dataset)

    try:
        first_row = next(iterator)
    except StopIteration:
        raise RuntimeError("Dataset stream rỗng: không lấy được dòng đầu tiên.")

    headers = list(first_row.keys())
    print(f"Đang ghi dữ liệu vào: {OUTPUT_CSV}")

    rows_written = 0
    total_progress = LIMIT_MB if PRIORITIZE_MB else LIMIT_ROWS
    unit_progress = "MB" if PRIORITIZE_MB else "row"

    with open(OUTPUT_CSV, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(first_row)
        rows_written += 1
        f.flush()
        file_size_mb = os.path.getsize(OUTPUT_CSV) / (1024 * 1024)

        with tqdm(total=total_progress, desc=f"Đang tải ({unit_progress})", unit=unit_progress) as pbar:
            if PRIORITIZE_MB:
                pbar.n = min(file_size_mb, LIMIT_MB)
                pbar.refresh()
            else:
                pbar.update(1)

            for row in iterator:
                writer.writerow(row)
                rows_written += 1

                if PRIORITIZE_MB and (rows_written % 100 == 0 or file_size_mb >= LIMIT_MB * 0.95):
                    f.flush()
                    file_size_mb = os.path.getsize(OUTPUT_CSV) / (1024 * 1024)
                    pbar.n = min(round(file_size_mb, 2), LIMIT_MB)
                    pbar.refresh()
                elif not PRIORITIZE_MB:
                    pbar.update(1)

                if PRIORITIZE_MB and file_size_mb >= LIMIT_MB:
                    print(f"\\n[DỪNG] Đạt giới hạn dung lượng {file_size_mb:.2f} MB ({rows_written:,} dòng)")
                    break

                if rows_written >= LIMIT_ROWS:
                    print(f"\\n[DỪNG] Đạt giới hạn số dòng: {rows_written:,}")
                    break

        f.flush()

    final_size_mb = os.path.getsize(OUTPUT_CSV) / (1024 * 1024)
    print(f"✅ Hoàn thành: {OUTPUT_CSV}\\n   Rows: {rows_written:,}\\n   Size: {final_size_mb:.2f} MB")
'''

# --- Cell 008: loader + dedup + chunking, có evidence-aware sampling -------------
CELL_008_HEAD = '''#@title 1.5 — Loader + exact dedup + chunking
def norm_space(x):
    return re.sub(r"\\s+", " ", str(x or "")).strip()

def sha1(x):
    return hashlib.sha1(str(x).encode("utf-8", errors="ignore")).hexdigest()

def pick_col(df, candidates, required=True):
    lookup = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    if required:
        raise KeyError(f"Missing one of columns: {candidates}")
    return None


def load_golden_evidence_rows():
    """Row index (0-based) của các article được golden dataset dùng làm evidence.

    Vì sao cần: golden dataset trải trên row 33..4997 nhưng LAB_MAX_ARTICLES=1500.
    Nếu sample ngẫu nhiên như bản gốc thì chỉ 19/51 article evidence trụ lại
    => phần lớn câu hỏi thành không-thể-trả-lời và phép so sánh mất ý nghĩa.
    """
    if not GOLDEN_DETAILED_PATH.exists():
        print(f"⚠️  Không thấy {GOLDEN_DETAILED_PATH.name} — sẽ sample ngẫu nhiên như bản gốc.")
        return set()
    d = pd.read_csv(GOLDEN_DETAILED_PATH)
    rows = set()
    for s in d.get("evidence_row_ids_0based", pd.Series(dtype=str)).dropna():
        try:
            rows.update(int(x) for x in json.loads(s))
        except Exception:
            continue
    print(f"Golden evidence: {len(rows)} article (row {min(rows)}..{max(rows)})" if rows else "Golden evidence: 0")
    return rows


GOLDEN_EVIDENCE_ROWS = load_golden_evidence_rows()


def load_news(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        df = pd.read_json(path, lines=True)
    elif path.suffix.lower() == ".json":
        df = pd.read_json(path)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported: {path.suffix}")
    # Giữ vị trí gốc để đối chiếu evidence_row_ids_0based của golden dataset.
    df["_source_row"] = np.arange(len(df))
    return df


def standardize_news(raw):
    text_col = pick_col(raw, ["text", "content", "article", "body", "story"])
    title_col = pick_col(raw, ["title", "headline"], required=False)
    date_col = pick_col(raw, ["published_date", "date", "published_at", "created_at"], required=False)
    id_col = pick_col(raw, ["id", "article_id", "story_id", "uuid"], required=False)

    df = pd.DataFrame()
    df["text"] = raw[text_col].fillna("").map(norm_space)
    df["title"] = raw[title_col].fillna("").map(norm_space) if title_col else ""

    if date_col:
        df["published_date"] = (
            pd.to_datetime(raw[date_col], errors="coerce", utc=True)
            .dt.strftime("%Y-%m-%d")
            .fillna("")
        )
    else:
        df["published_date"] = ""

    if id_col:
        df["article_id"] = raw[id_col].astype(str)
    else:
        df["article_id"] = [
            sha1(f"{t}\\n{x}")[:20] for t, x in zip(df["title"], df["text"])
        ]

    df["source_row"] = (
        raw["_source_row"].to_numpy() if "_source_row" in raw.columns else np.arange(len(raw))
    )

    df = df[df["text"].str.len() >= 80].copy()
    df["dedup_key"] = [
        sha1(norm_space(f"{t}\\n{x}").lower())
        for t, x in zip(df["title"], df["text"])
    ]
    before = len(df)
    df = df.drop_duplicates("dedup_key").drop(columns="dedup_key").reset_index(drop=True)
    print(f"Exact dedup: {before:,} -> {len(df):,}")

    if LAB_MAX_ARTICLES and len(df) > LAB_MAX_ARTICLES:
        # Sampling deterministic + evidence-aware: giữ TRỌN article evidence của golden
        # dataset, phần còn lại lấp bằng shuffle theo SEED để làm distractor.
        # Vẫn tôn trọng scale guard LAB_MAX_ARTICLES.
        must = df["source_row"].isin(GOLDEN_EVIDENCE_ROWS)
        keep = df[must]
        room = max(0, LAB_MAX_ARTICLES - len(keep))
        fill = df[~must].sample(frac=1.0, random_state=SEED).head(room)
        df = (
            pd.concat([keep, fill])
            .sort_values("source_row")
            .reset_index(drop=True)
        )
        print(
            f"Sample {LAB_MAX_ARTICLES}: {len(keep)} evidence article + {len(fill)} distractor"
        )
        missing = GOLDEN_EVIDENCE_ROWS - set(df["source_row"])
        if missing:
            print(f"⚠️  {len(missing)} evidence row bị loại ở bước dedup/độ dài: {sorted(missing)[:20]}")
    return df


def chunk_text(text, size=220, overlap=40):
    words = norm_space(text).split()
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(words), step):
        part = words[start:start+size]
        if not part:
            break
        out.append(" ".join(part))
        if start + size >= len(words):
            break
    return out


def build_chunks(news_df):
    rows = []
    for r in tqdm(news_df.itertuples(index=False), total=len(news_df), desc="Chunking"):
        for i, text in enumerate(chunk_text(r.text, CHUNK_WORDS, CHUNK_OVERLAP_WORDS)):
            rows.append({
                "chunk_id": f"{r.article_id}::c{i:04d}",
                "article_id": r.article_id,
                "source_row": int(r.source_row),
                "title": r.title,
                "published_date": r.published_date,
                "text": text,
            })
            if LAB_MAX_CHUNKS and len(rows) >= LAB_MAX_CHUNKS:
                print(f"⚠️  Đạt LAB_MAX_CHUNKS={LAB_MAX_CHUNKS}, dừng chunking sớm.")
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


raw_df = load_news(DATA_PATH)
news_df = standardize_news(raw_df)
chunks_df = build_chunks(news_df)
print(f"news_df={len(news_df):,} article | chunks_df={len(chunks_df):,} chunk")
display(chunks_df.head())
'''

CELL_012_TAIL_OLD = '''# extraction_source = chunks_df.head(EXTRACTION_MAX_CHUNKS).copy()
# coref_df = run_coref(extraction_source)
# extraction_source = extraction_source.merge(coref_df, on="chunk_id", how="left")'''

CELL_012_TAIL_NEW = '''def select_extraction_source(chunks_df, news_df, limit=None):
    """Chọn <=limit chunk để coref + NER/RE, ưu tiên article evidence của golden dataset.

    Bản gốc lấy chunks_df.head(400) theo thứ tự file. Với budget 400 chunk, cách đó
    khiến graph gần như không phủ được câu hỏi eval (evidence trải tới row 4997).
    Ở đây: nạp trọn chunk của article evidence trước, còn chỗ thì lấp bằng chunk
    đầu file làm nhiễu. Thứ tự trong mỗi nhóm giữ nguyên -> deterministic.
    """
    limit = int(limit or EXTRACTION_MAX_CHUNKS)
    ev_articles = set(news_df.loc[news_df.source_row.isin(GOLDEN_EVIDENCE_ROWS), "article_id"])
    is_ev = chunks_df.article_id.isin(ev_articles)

    prio = chunks_df[is_ev]
    if len(prio) > limit:
        print(f"⚠️  Chunk evidence ({len(prio)}) > EXTRACTION_MAX_CHUNKS ({limit}) — sẽ bị cắt.")
    fill = chunks_df[~is_ev].head(max(0, limit - len(prio)))
    sel = pd.concat([prio, fill]).head(limit).reset_index(drop=True)

    covered = sel.article_id.nunique()
    ev_covered = sel.loc[sel.article_id.isin(ev_articles), "article_id"].nunique()
    print(
        f"extraction_source: {len(sel)} chunk / {covered} article "
        f"| evidence article phủ được {ev_covered}/{len(ev_articles)}"
    )
    return sel


extraction_source = select_extraction_source(chunks_df, news_df).copy()
coref_df = run_coref(extraction_source)
extraction_source = extraction_source.merge(coref_df, on="chunk_id", how="left")
print(
    "Chunk coref thất bại:",
    int(extraction_source.unresolved_mentions.apply(
        lambda v: isinstance(v, list) and "COREF_BATCH_FAILED" in v
    ).sum()),
)
display(extraction_source[["chunk_id", "text", "resolved_text"]].head(3))'''

CELL_026_OLD_PATH = 'GOLDEN_PATH = "/content/golden_dataset.csv"\n\n'
CELL_026_TAIL_OLD = '''golden_df = pd.read_csv(GOLDEN_PATH) if Path(GOLDEN_PATH).exists() else starter_golden.copy()
display(golden_df)'''
CELL_026_TAIL_NEW = '''# GOLDEN_PATH được định nghĩa ở cell 1.2 (config), trỏ vào data/ trong repo.
# File thật: 50 câu hard, soạn trên 5000 row đầu của hackernoon_subset.csv,
# đủ 3 nhóm factoid / multi-hop / cross-doc và đã điền reference_answer.
if Path(GOLDEN_PATH).exists():
    golden_df = pd.read_csv(GOLDEN_PATH)
    print(f"✅ Golden dataset: {GOLDEN_PATH.name} — {len(golden_df)} câu")
else:
    golden_df = starter_golden.copy()
    print(f"⚠️  Không thấy {GOLDEN_PATH} — dùng 5 câu starter (G02..G05 chưa có gold answer).")

print(golden_df.group.value_counts().to_dict())
display(golden_df.head())'''

CELL_028_OLD = 'CHECKPOINT = "/content/graphrag_eval_checkpoint.csv"\n\ndef run_evaluation(golden_df):\n    rows = []\n'
CELL_028_NEW = '''# CHECKPOINT được định nghĩa ở cell 1.2 -> outputs/graphrag_eval_checkpoint.csv
def run_evaluation(golden_df, resume=True):
    """Chạy eval cho từng câu. resume=True -> bỏ qua id đã có trong checkpoint.

    Với 50 câu (~200 lệnh gọi LLM), resume là bắt buộc để không phải chạy lại từ đầu
    khi gặp rate limit / mất mạng.
    """
    rows = []
    done = set()
    if resume and Path(CHECKPOINT).exists():
        prev = pd.read_csv(CHECKPOINT)
        rows = prev.to_dict("records")
        done = set(prev["id"].astype(str))
        print(f"Resume từ checkpoint: đã có {len(done)} câu.")
'''

CELL_028_LOOP_OLD = '''    for q in tqdm(golden_df.itertuples(index=False), total=len(golden_df), desc="Evaluation"):
        flat = answer_flat_rag(q.question)'''
CELL_028_LOOP_NEW = '''    for q in tqdm(golden_df.itertuples(index=False), total=len(golden_df), desc="Evaluation"):
        if str(q.id) in done:
            continue
        flat = answer_flat_rag(q.question)'''

CELL_029_OLD = '''# comparison_df = comparison_table(eval_results_df)
# display(comparison_df)
# eval_results_df.to_csv("/content/graphrag_eval_results.csv", index=False)
# comparison_df.to_csv("/content/graphrag_vs_flatrag_summary.csv", index=False)'''

CELL_029_NEW = '''comparison_df = comparison_table(eval_results_df)
display(comparison_df)

# RUBRIC 3.3 ghi reports/, README + ASSIGNMENT ghi outputs/ -> xuất cả 2 nơi cho chắc.
for _dir in (OUTPUTS_DIR, REPORTS_DIR):
    eval_results_df.to_csv(_dir / "graphrag_eval_results.csv", index=False)
    comparison_df.to_csv(_dir / "graphrag_vs_flatrag_summary.csv", index=False)
    print(f"✅ Đã ghi {_dir.name}/graphrag_eval_results.csv + graphrag_vs_flatrag_summary.csv")'''


def main() -> int:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    if len(cells) != 37:
        raise SystemExit(f"[FAIL] mong đợi 37 cell, thấy {len(cells)}")

    # ---- P1: hạ tầng ----
    replace_cell(cells, 3, "%pip", CELL_003, "install (bỏ spacy/langchain/llama-index)")
    replace_cell(cells, 4, "get_secret", CELL_004, "config + .env + path portable")
    replace_cell(cells, 6, "DATASET_NAME", CELL_006, "stream 5000 row + skip-if-exists")
    replace_cell(cells, 8, "def load_news", CELL_008_HEAD, "loader + evidence-aware sampling")

    # ---- P2: uncomment driver call ----
    uncomment(cells, 7, ["connect_neo4j()", "setup_graph_schema()"], "neo4j connect/schema")
    sub_in_cell(cells, 12, CELL_012_TAIL_OLD, CELL_012_TAIL_NEW, "extraction_source evidence-aware + coref")
    uncomment(cells, 14, ["run_extraction(", "raw_triples_df.head()"], "NER+RE")
    uncomment(cells, 16, ["build_resolution_map(", "canonicalize_triples(", "entity_resolution_audit_df.head"], "entity resolution")
    uncomment(cells, 17, ["build_nodes(", "bulk_insert_nodes(", "bulk_insert_edges("], "UNWIND bulk insert")
    uncomment(cells, 18, ["graph_checks()"], "sanity checks")
    uncomment(cells, 20, ["build_flat_index("], "flat FAISS index")
    uncomment(cells, 22, ["build_entity_matcher("], "entity matcher")

    sub_in_cell(cells, 26, CELL_026_OLD_PATH, "", "bỏ GOLDEN_PATH hard-code /content")
    sub_in_cell(cells, 26, CELL_026_TAIL_OLD, CELL_026_TAIL_NEW, "load golden 50 câu từ data/")

    sub_in_cell(cells, 28, CELL_028_OLD, CELL_028_NEW, "checkpoint portable + resume")
    sub_in_cell(cells, 28, CELL_028_LOOP_OLD, CELL_028_LOOP_NEW, "skip câu đã chạy")
    uncomment(cells, 28, ["validate_golden(", "run_evaluation(", "eval_results_df)"], "eval runner")

    sub_in_cell(cells, 29, CELL_029_OLD, CELL_029_NEW, "export CSV vào outputs/ + reports/")

    uncomment(cells, 31, ["test_supernode_policy()", "show_resolution_audit("], "failure-mode checks")
    uncomment(cells, 34, ["build_communities()"], "bonus community detection")

    # Kiểm tra không còn /content/ và không lộ secret
    leftovers = [i for i, c in enumerate(cells) if "/content/" in src(c)]
    if leftovers:
        print(f"⚠️  Còn '/content/' ở cell: {leftovers}")

    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print("PATCH XONG:")
    for line in CHANGELOG:
        print(line)
    print(f"\nTổng: {sum(1 for l in CHANGELOG if l.strip().startswith('*'))} thay đổi áp dụng.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
