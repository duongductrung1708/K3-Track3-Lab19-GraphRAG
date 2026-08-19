"""Patch vòng 2 cho notebook — sau khi chạy thật lần 1 và phát hiện 3 vấn đề đo được.

Bối cảnh (số liệu thật của run 1, 2026-08-19):
  - Dataset thật `HackerNoon/tech-company-news-data-dump` có cột body là `description`,
    và nó chỉ là TEASER bị cắt (~208 ký tự / ~32 từ, kết thúc ". . ."). Cột `title`
    (~71 ký tự) mới chứa sự kiện chính ("Aeris to acquire IoT business from Ericsson")
    nhưng notebook KHÔNG đưa title vào chunk text -> extractor không bao giờ thấy.
    Kết quả: 400 chunk -> chỉ 113 node / 79 edge, max degree 8, audit 1 dòng.
  - Groq free tier: TPD 200k / model. coref+extraction đã tiêu hết TPD của
    `openai/gpt-oss-120b` -> cell 028 (eval) chết 429 giữa đường, mất trắng công.
  - Không có cache: mọi lần chạy lại phải trả lại toàn bộ token cho coref+extraction.

5 patch (idempotent, assert anchor — anchor lệch thì raise chứ không patch sai chỗ):
  P1 cell 004  thêm cache path, pool model fallback, EVAL_MAX_PER_GROUP
  P2 cell 008  đưa `title` vào text bài báo (áp dụng cho CẢ flat RAG lẫn GraphRAG
               nên phép so sánh vẫn công bằng: cùng một corpus)
  P3 cell 010  gặp 429 tokens-per-day -> chuyển sang model kế tiếp trong pool (sticky)
  P4 cell 012/014 cache tăng dần ra đĩa: mất mạng / hết TPD giữa đường vẫn giữ được
               phần đã làm, lần sau chỉ chạy phần còn thiếu
  P5 cell 028  eval theo subset phân tầng theo `group`, in rõ lấy gì / bỏ gì
               (không cắt ngầm), vì 50 câu x ~17k token vượt xa TPD free tier

Chạy: .venv/Scripts/python.exe tools/patch_run2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb"


def patch(cells, idx, old, new, label):
    src = cells[idx].source
    if new in src:
        print(f"  [skip] {label} — đã patch")
        return 0
    assert old in src, f"{label}: KHÔNG thấy anchor trong cell {idx}"
    cells[idx].source = src.replace(old, new, 1)
    print(f"  [ok]   {label}")
    return 1


def main() -> int:
    nb = nbformat.read(NB, as_version=4)
    c = nb.cells
    n = 0

    # ---------------- P1: config ----------------
    n += patch(c, 4,
        'CHECKPOINT = OUTPUTS_DIR / "graphrag_eval_checkpoint.csv"',
        '''CHECKPOINT = OUTPUTS_DIR / "graphrag_eval_checkpoint.csv"

# Cache 2 chặng LLM đắt nhất (coref ~989s, NER+RE ~1091s ở run 1). Groq free tier
# giới hạn 200k token/ngày/model nên chạy lại từ đầu là không khả thi -> ghi ra đĩa
# theo từng batch, lần sau chỉ làm phần còn thiếu.
COREF_CACHE = OUTPUTS_DIR / "cache_coref.csv"
TRIPLES_CACHE = OUTPUTS_DIR / "cache_raw_triples.csv"''',
        "P1a cache path")

    n += patch(c, 4,
        'HF_TOKEN = get_secret("HF_TOKEN", "")',
        '''HF_TOKEN = get_secret("HF_TOKEN", "")

# Pool model dự phòng: TPD (tokens-per-day) của Groq tính RIÊNG theo từng model.
# Hết quota model chính -> groq_chat() tự chuyển sang model kế tiếp thay vì chết 429
# giữa pipeline. Ghi lại model nào thực sự phục vụ để báo cáo trung thực.
GROQ_MODEL_FALLBACKS = [
    m.strip() for m in str(get_secret("GROQ_MODEL_FALLBACKS", "")).split(",") if m.strip()
]

# 50 câu golden x ~17k token/câu (2 answer + 2 judge) vượt xa TPD free tier.
# None = chạy tất cả; số n = lấy n câu ĐẦU của mỗi group (factoid/multi-hop/cross-doc).
_emg = str(get_secret("EVAL_MAX_PER_GROUP", "") or "").strip()
EVAL_MAX_PER_GROUP = int(_emg) if _emg.isdigit() else None''',
        "P1b model pool + eval subset")

    # ---------------- P2: title vào text ----------------
    n += patch(c, 8,
        '''    df = pd.DataFrame()
    df["text"] = raw[text_col].fillna("").map(norm_space)
    df["title"] = raw[title_col].fillna("").map(norm_space) if title_col else ""''',
        '''    df = pd.DataFrame()
    body = raw[text_col].fillna("").map(norm_space)
    df["title"] = raw[title_col].fillna("").map(norm_space) if title_col else ""

    # Dataset thật: `description` chỉ là teaser bị cắt (~32 từ, kết thúc ". . ."),
    # còn `title` mới chứa sự kiện chính ("Aeris to acquire IoT business from Ericsson").
    # Run 1 không ghép title => extractor mù với hầu hết quan hệ: 400 chunk -> 79 edge.
    # Ghép title vào đầu body (title là một phần của bài báo, không phải bịa thêm).
    # Áp dụng cho cả Flat RAG lẫn GraphRAG vì cùng dùng chunks_df -> so sánh vẫn công bằng.
    if title_col:
        df["text"] = [
            b if (not t or b.lower().startswith(t.lower())) else (f"{t}. {b}" if b else t)
            for t, b in zip(df["title"], body)
        ]
    else:
        df["text"] = body''',
        "P2 title vào text")

    # ---------------- P3: model fallback khi 429 TPD ----------------
    n += patch(c, 10,
        '''def groq_chat(messages, model=None, json_mode=False, max_retries=4):
    if groq_client is None:
        raise RuntimeError("Thiếu GROQ_API_KEY.")
    model = model or GROQ_MODEL
    if not model:
        raise RuntimeError("Thiếu GROQ_MODEL.")

    last = None
    for attempt in range(max_retries):''',
        '''# Model đã hết TPD trong phiên này -> không thử lại nữa (sticky).
TPD_EXHAUSTED = set()
MODEL_CALLS = Counter()          # model nào thực sự phục vụ bao nhiêu call

def _is_tpd_error(e) -> bool:
    s = str(e)
    return "429" in s and ("per day" in s or "TPD" in s)

def _model_chain(model):
    """Model chính + các fallback, bỏ model đã hết TPD."""
    chain = [model] + [m for m in GROQ_MODEL_FALLBACKS if m != model]
    live = [m for m in chain if m not in TPD_EXHAUSTED]
    return live or chain[-1:]        # hết sạch thì vẫn thử cái cuối để lộ lỗi thật

def groq_chat(messages, model=None, json_mode=False, max_retries=4):
    if groq_client is None:
        raise RuntimeError("Thiếu GROQ_API_KEY.")
    model = model or GROQ_MODEL
    if not model:
        raise RuntimeError("Thiếu GROQ_MODEL.")

    last = None
    for _m in _model_chain(model):
        try:
            return _groq_chat_one(messages, _m, json_mode, max_retries)
        except Exception as e:
            last = e
            if not _is_tpd_error(e):
                raise
            TPD_EXHAUSTED.add(_m)
            print(f"⚠️  {_m} hết TPD -> chuyển model fallback.", flush=True)
    raise RuntimeError(last)

def _groq_chat_one(messages, model, json_mode=False, max_retries=4):
    last = None
    for attempt in range(max_retries):''',
        "P3a model fallback")

    n += patch(c, 10,
        '''            resp = groq_client.chat.completions.create(**kwargs)
            usage = {}''',
        '''            resp = groq_client.chat.completions.create(**kwargs)
            MODEL_CALLS[model] += 1
            usage = {}''',
        "P3b đếm call theo model")

    n += patch(c, 10,
        '''        except Exception as e:
            last = e
            if attempt == max_retries - 1:
                break
            time.sleep(min(20, 2**attempt + random.random()))
    raise RuntimeError(last)''',
        '''        except Exception as e:
            last = e
            if _is_tpd_error(e):
                raise                      # để groq_chat() đổi model, đừng ngủ vô ích
            if attempt == max_retries - 1:
                break
            time.sleep(min(20, 2**attempt + random.random()))
    raise RuntimeError(last)''',
        "P3c không retry vô ích khi hết TPD")

    # ---------------- P4a: cache coref ----------------
    n += patch(c, 12,
        '''def run_coref(chunks_subset, batch_size=5):
    out = []
    for start in tqdm(range(0, len(chunks_subset), batch_size), desc="Coref"):
        batch = chunks_subset.iloc[start:start+batch_size]
        try:
            df, _ = resolve_coref_batch(batch)
        except Exception:
            df = pd.DataFrame({
                "chunk_id": batch["chunk_id"].tolist(),
                "resolved_text": batch["text"].tolist(),
                "unresolved_mentions": [["COREF_BATCH_FAILED"] for _ in range(len(batch))],
            })
        out.append(df)
    return pd.concat(out, ignore_index=True)''',
        '''def run_coref(chunks_subset, batch_size=5, cache_path=None):
    """Coref theo batch, GHI CACHE SAU MỖI BATCH.

    Vì sao: run 1 mất 989s và tiêu phần lớn TPD 200k/ngày. Hết quota giữa đường mà
    không cache thì lần sau phải trả lại toàn bộ token cho phần đã làm xong.
    """
    cache_path = Path(cache_path or COREF_CACHE)
    out, done = [], set()
    if cache_path.exists():
        prev = pd.read_csv(cache_path)
        prev = prev[prev.chunk_id.isin(set(chunks_subset.chunk_id))]
        if len(prev):
            out.append(prev)
            done = set(prev.chunk_id.astype(str))
            print(f"Cache coref: dùng lại {len(done)}/{len(chunks_subset)} chunk.")

    todo = chunks_subset[~chunks_subset.chunk_id.astype(str).isin(done)]
    for start in tqdm(range(0, len(todo), batch_size), desc="Coref"):
        batch = todo.iloc[start:start+batch_size]
        try:
            df, _ = resolve_coref_batch(batch)
        except Exception:
            df = pd.DataFrame({
                "chunk_id": batch["chunk_id"].tolist(),
                "resolved_text": batch["text"].tolist(),
                "unresolved_mentions": [["COREF_BATCH_FAILED"] for _ in range(len(batch))],
            })
        out.append(df)
        pd.concat(out, ignore_index=True).to_csv(cache_path, index=False)
    return pd.concat(out, ignore_index=True)''',
        "P4a cache coref")

    # ---------------- P4b: cache extraction ----------------
    n += patch(c, 14,
        '''def run_extraction(source_df, batch_size=4):
    meta = source_df.set_index("chunk_id")["published_date"].to_dict()
    triples, errors = [], []

    for start in tqdm(range(0, len(source_df), batch_size), desc="NER+RE"):
        batch = source_df.iloc[start:start+batch_size]''',
        '''def run_extraction(source_df, batch_size=4, cache_path=None):
    """NER+RE theo batch, GHI CACHE SAU MỖI BATCH (xem lý do ở run_coref)."""
    cache_path = Path(cache_path or TRIPLES_CACHE)
    meta = source_df.set_index("chunk_id")["published_date"].to_dict()
    triples, errors = [], []
    done_chunks = set()

    if cache_path.exists():
        prev = pd.read_csv(cache_path)
        prev = prev[prev.source_chunk_id.isin(set(source_df.chunk_id))]
        if len(prev):
            triples = prev.to_dict("records")
            done_chunks = set(prev.source_chunk_id.astype(str))
            print(f"Cache NER+RE: dùng lại {len(triples)} triple "
                  f"từ {len(done_chunks)} chunk đã xử lý.")

    # chunk đã có triple thì bỏ qua; chunk không sinh triple nào vẫn phải chạy lại
    # (không phân biệt được "đã chạy, 0 triple" với "chưa chạy") -> chấp nhận, an toàn hơn.
    remaining = source_df[~source_df.chunk_id.astype(str).isin(done_chunks)]
    for start in tqdm(range(0, len(remaining), batch_size), desc="NER+RE"):
        batch = remaining.iloc[start:start+batch_size]''',
        "P4b cache extraction — vào hàm")

    n += patch(c, 14,
        '''                    "confidence": float(x.get("confidence") or 0.0),
                })

    return pd.DataFrame(triples), pd.DataFrame(errors)''',
        '''                    "confidence": float(x.get("confidence") or 0.0),
                })

        pd.DataFrame(triples).to_csv(cache_path, index=False)   # checkpoint mỗi batch

    return pd.DataFrame(triples), pd.DataFrame(errors)''',
        "P4c cache extraction — ghi mỗi batch")

    # ---------------- P5: eval subset phân tầng ----------------
    n += patch(c, 28,
        '''validate_golden(golden_df, require_answers=True)
eval_results_df = run_evaluation(golden_df)''',
        '''def select_eval_subset(golden_df, max_per_group=None):
    """Lấy subset phân tầng theo `group`, IN RÕ lấy gì / bỏ gì.

    50 câu x ~17k token (flat answer + graph answer + 2 lượt judge với context 18k ký tự)
    vượt xa TPD 200k/ngày/model của Groq free tier. Cắt thì phải cắt công khai và
    cân theo group, không lấy head(n) làm lệch tỉ lệ factoid/multi-hop/cross-doc.
    """
    if not max_per_group:
        print(f"Eval: chạy toàn bộ {len(golden_df)} câu.")
        return golden_df
    parts = [g.head(max_per_group) for _, g in golden_df.groupby("group", sort=True)]
    sub = pd.concat(parts).sort_index()
    kept = sub.group.value_counts().to_dict()
    full = golden_df.group.value_counts().to_dict()
    print(f"⚠️  Eval subset {len(sub)}/{len(golden_df)} câu "
          f"(EVAL_MAX_PER_GROUP={max_per_group}, lý do: TPD Groq free tier).")
    for grp in sorted(full):
        print(f"     {grp:12s} lấy {kept.get(grp,0):2d}/{full[grp]:2d} "
              f"-> bỏ {full[grp]-kept.get(grp,0)}")
    return sub

validate_golden(golden_df, require_answers=True)
eval_golden_df = select_eval_subset(golden_df, EVAL_MAX_PER_GROUP)
eval_results_df = run_evaluation(eval_golden_df)
print("Model đã phục vụ:", dict(MODEL_CALLS))''',
        "P5 eval subset phân tầng")

    nbformat.write(nb, NB)
    print(f"\nĐã áp dụng {n} patch -> {NB.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
