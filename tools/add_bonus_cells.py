"""Thêm 3 cell BONUS vào notebook (Challenge A + Bonus B bước 4-5 + Bonus C driver).

Chạy SAU `tools/patch_notebook.py`:
    .venv/Scripts/python.exe tools/patch_notebook.py
    .venv/Scripts/python.exe tools/add_bonus_cells.py

Vì sao tách khỏi patch_notebook.py: script kia thao tác theo *index cell cố định* và
assert đúng 37 cell. Chèn cell mới làm lệch index -> phải là bước riêng, chạy sau.

Idempotent: mỗi cell chèn vào mang một marker `# @bonus:<tên>` ở dòng đầu; thấy marker
rồi thì bỏ qua. Anchor tìm theo NỘI DUNG (không theo index) nên không lệ thuộc thứ tự.

Ba khoảng trống được lấp (đều là yêu cầu tính điểm bonus nhưng notebook gốc bỏ trống):
  A. Challenge A near-dedup: notebook gốc KHÔNG có dòng code nào -> viết MinHash+LSH.
  B. Bonus B: cell 34 chỉ làm tới bước 3 (write community_id), thiếu bước 4-5
     (LLM summarize community + global search trên report).
  C. Bonus C: cell 35 định nghĩa hàm nhưng KHÔNG cell nào gọi -> không có output để chấm.
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


def new_code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def insert_after(cells: list, anchor: str, marker: str, text: str, label: str) -> None:
    """Chèn code cell ngay sau cell đầu tiên chứa `anchor`. Bỏ qua nếu đã có `marker`."""
    if any(marker in src(c) for c in cells):
        CHANGELOG.append(f"  = {label} (đã có, bỏ qua)")
        return
    pos = next((i for i, c in enumerate(cells) if anchor in src(c)), None)
    if pos is None:
        raise SystemExit(f"[FAIL] {label}: không thấy anchor {anchor!r}")
    cells.insert(pos + 1, new_code_cell(text))
    CHANGELOG.append(f"  * {label} -> chèn sau cell {pos}")


# ---------------------------------------------------------------------------
# A. Challenge A — Near-duplicate detection bằng MinHash + LSH banding
# ---------------------------------------------------------------------------

CELL_NEAR_DEDUP = '''# @bonus:near_dedup
#@title 1.5b — Challenge A: Near-dedup bằng MinHash + LSH (KHÔNG O(N²))
# Exact sha1 ở cell 1.5 chỉ bắt bài trùng byte-for-byte. Tech news bị repost qua nhiều
# outlet với title/boilerplate khác nhau -> cần near-dup.
#
# Thiết kế (và 3 điều Challenge A yêu cầu nêu trong báo cáo):
#   1. THRESHOLD: LSH banding b=32, r=4 -> ngưỡng ứng viên ~ (1/b)^(1/r) = 0.42.
#      Kết luận near-dup chỉ khi Jaccard THẬT >= NEAR_DUP_THRESHOLD = 0.80.
#   2. FALSE POSITIVE: mọi cặp ứng viên đều được verify Jaccard thật trên tập shingle;
#      cặp có true_jaccard < 0.80 bị đánh REJECT_BELOW_THRESHOLD và đếm thành FP rate.
#   3. AUDIT: near_dup_audit_df lưu từng cặp (title, est/true jaccard, decision).
#
# Vì sao không O(N²): chỉ so sánh các cặp RƠI CÙNG BUCKET của ít nhất 1 band.
# Chi phí = O(N * #shingle) cho signature + O(#candidate) cho verify.
import zlib

NEAR_DUP_NUM_PERM = 128
NEAR_DUP_BANDS = 32
NEAR_DUP_ROWS = NEAR_DUP_NUM_PERM // NEAR_DUP_BANDS      # = 4
NEAR_DUP_THRESHOLD = 0.80
NEAR_DUP_SHINGLE_K = 5
NEAR_DUP_MAX_BUCKET = 200        # bucket to hơn -> bỏ qua để không tụt về O(m²)
NEAR_DUP_APPLY = True            # True = thực sự loại near-dup rồi chunk lại

_ND_PRIME = (1 << 31) - 1
_nd_rng = np.random.default_rng(SEED)
_ND_A = _nd_rng.integers(1, _ND_PRIME, size=NEAR_DUP_NUM_PERM, dtype=np.uint64)
_ND_B = _nd_rng.integers(0, _ND_PRIME, size=NEAR_DUP_NUM_PERM, dtype=np.uint64)


def nd_shingles(text, k=NEAR_DUP_SHINGLE_K):
    """Tập shingle k-gram theo TỪ. Bỏ dấu câu để repost đổi punctuation vẫn khớp."""
    words = re.sub(r"[^\\w\\s]", " ", norm_space(text).lower()).split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def nd_signature(shingle_set):
    """MinHash signature. Xử lý theo block để không cấp phát ma trận khổng lồ."""
    out = np.full(NEAR_DUP_NUM_PERM, _ND_PRIME, dtype=np.uint64)
    if not shingle_set:
        return out
    xs = np.fromiter(
        (zlib.crc32(s.encode("utf-8")) for s in shingle_set),
        dtype=np.uint64, count=len(shingle_set),
    )
    for i in range(0, len(xs), 4096):
        blk = xs[i:i + 4096]
        h = (_ND_A[:, None] * blk[None, :] + _ND_B[:, None]) % _ND_PRIME
        np.minimum(out, h.min(axis=1), out=out)
    return out


class _NdUnionFind:
    def __init__(self, n):
        self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def find_near_duplicates(df):
    """-> (audit_df, groups). groups = list các list index near-dup cùng nhóm."""
    shingle_sets = [
        nd_shingles(f"{t} {x}")
        for t, x in tqdm(zip(df.title, df.text), total=len(df), desc="Shingling")
    ]
    sigs = np.vstack([nd_signature(s) for s in shingle_sets])

    # --- LSH banding ---
    buckets = defaultdict(list)
    for idx in range(len(df)):
        for b in range(NEAR_DUP_BANDS):
            band = sigs[idx, b * NEAR_DUP_ROWS:(b + 1) * NEAR_DUP_ROWS].tobytes()
            buckets[(b, band)].append(idx)

    candidates, skipped_buckets = set(), 0
    for members in buckets.values():
        if len(members) < 2:
            continue
        if len(members) > NEAR_DUP_MAX_BUCKET:
            skipped_buckets += 1
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                candidates.add((members[i], members[j]))

    lsh_threshold = (1.0 / NEAR_DUP_BANDS) ** (1.0 / NEAR_DUP_ROWS)
    print(
        f"LSH: {len(buckets):,} bucket -> {len(candidates):,} cặp ứng viên "
        f"(so với {len(df)*(len(df)-1)//2:,} cặp nếu brute-force O(N²)); "
        f"ngưỡng banding ~{lsh_threshold:.2f}; bucket bị bỏ vì quá lớn: {skipped_buckets}"
    )

    # --- Verify Jaccard thật, chỉ trên cặp ứng viên ---
    uf = _NdUnionFind(len(df))
    rows = []
    for i, j in sorted(candidates):
        est = float((sigs[i] == sigs[j]).mean())
        a, b_ = shingle_sets[i], shingle_sets[j]
        union = len(a | b_)
        true_j = (len(a & b_) / union) if union else 0.0
        is_dup = true_j >= NEAR_DUP_THRESHOLD
        if is_dup:
            uf.union(i, j)
        rows.append({
            "left_row": int(df.source_row.iloc[i]),
            "right_row": int(df.source_row.iloc[j]),
            "left_title": df.title.iloc[i][:90],
            "right_title": df.title.iloc[j][:90],
            "est_jaccard": round(est, 4),
            "true_jaccard": round(true_j, 4),
            "decision": "NEAR_DUP" if is_dup else "REJECT_BELOW_THRESHOLD",
        })

    audit_df = pd.DataFrame(rows)
    if not audit_df.empty:
        fp = int((audit_df.decision == "REJECT_BELOW_THRESHOLD").sum())
        print(
            f"Verify: {len(audit_df)} cặp | NEAR_DUP={len(audit_df)-fp} | "
            f"false positive của LSH={fp} ({fp/len(audit_df):.1%}) "
            f"-> đã bị chặn bởi ngưỡng Jaccard thật {NEAR_DUP_THRESHOLD}"
        )
    else:
        print("Verify: không có cặp ứng viên nào.")

    groups = defaultdict(list)
    for i in range(len(df)):
        groups[uf.find(i)].append(i)
    return audit_df, [g for g in groups.values() if len(g) > 1]


near_dup_audit_df, near_dup_groups = find_near_duplicates(news_df)
display(
    near_dup_audit_df.sort_values("true_jaccard", ascending=False).head(15)
    if not near_dup_audit_df.empty else near_dup_audit_df
)

if NEAR_DUP_APPLY and near_dup_groups:
    # Quy tắc an toàn: KHÔNG BAO GIỜ loại article là evidence của golden dataset
    # (loại đi là câu hỏi eval mất căn cứ). Chỉ loại bản trùng KHÔNG phải evidence.
    ev_rows = set(GOLDEN_EVIDENCE_ROWS)
    drop_idx, kept_ev_dups = [], 0
    for g in near_dup_groups:
        ev_members = [i for i in g if int(news_df.source_row.iloc[i]) in ev_rows]
        if len(ev_members) > 1:
            kept_ev_dups += len(ev_members) - 1
        keep = ev_members[0] if ev_members else min(
            g, key=lambda i: int(news_df.source_row.iloc[i])
        )
        drop_idx += [i for i in g if i != keep and i not in ev_members]

    before = len(news_df)
    news_df = news_df.drop(news_df.index[drop_idx]).reset_index(drop=True)
    print(
        f"Near-dedup: {before:,} -> {len(news_df):,} article "
        f"(loại {len(drop_idx)}; giữ lại {kept_ev_dups} bản trùng vì là evidence golden)"
    )
    chunks_df = build_chunks(news_df)
    print(f"Chunk lại sau near-dedup: chunks_df={len(chunks_df):,}")
else:
    print("Không có nhóm near-dup nào để loại (hoặc NEAR_DUP_APPLY=False).")
'''

# ---------------------------------------------------------------------------
# B. Bonus B bước 4-5 — LLM summarize community + global search
# ---------------------------------------------------------------------------

CELL_COMMUNITY_REPORTS = '''# @bonus:community_reports
#@title Bonus B (bước 4-5) — Community report + Global Search
# Cell trước mới làm tới bước 3 của Bonus B (export edges -> NetworkX -> ghi community_id).
# Còn thiếu bước 4 (LLM summarize community) và bước 5 (query global trên report).
COMMUNITY_MIN_SIZE = 3
COMMUNITY_MAX_REPORTS = 12
COMMUNITY_EDGES_PER_REPORT = 60

COMMUNITY_SYSTEM = """
You summarize a community of entities from a tech-news knowledge graph.
Use only the supplied relations. Do not invent facts.
Return strict JSON only.
""".strip()


def summarize_community(cid):
    edges = run_cypher("""
    MATCH (a:Entity {community_id:$cid})-[r]->(b:Entity)
    RETURN a.name AS source, type(r) AS relation, b.name AS target,
           r.published_date AS published_date, r.evidence AS evidence
    ORDER BY coalesce(r.published_date,'') DESC
    LIMIT $limit
    """, cid=int(cid), limit=int(COMMUNITY_EDGES_PER_REPORT))
    if not edges:
        return None

    lines = [
        f"{e['source']} -{e['relation']}-> {e['target']} "
        f"| date={e.get('published_date') or 'unknown'}"
        + (f" | {norm_space(e['evidence'])[:160]}" if e.get("evidence") else "")
        for e in edges
    ]
    obj, _ = groq_json(
        COMMUNITY_SYSTEM,
        "RELATIONS:\\n" + "\\n".join(lines) + """

Return:
{"title":"short community name","summary":"4-6 sentences","key_entities":["..."]}""",
    )
    return {
        "community_id": int(cid),
        "title": norm_space(obj.get("title")),
        "summary": norm_space(obj.get("summary")),
        "key_entities": ", ".join(obj.get("key_entities") or [])[:300],
        "n_edges": len(edges),
    }


def build_community_reports(community_df):
    sizes = community_df.community_id.value_counts()
    cids = [int(c) for c in sizes[sizes >= COMMUNITY_MIN_SIZE].index[:COMMUNITY_MAX_REPORTS]]
    print(f"Community >= {COMMUNITY_MIN_SIZE} node: {len(sizes[sizes >= COMMUNITY_MIN_SIZE])}"
          f" -> summarize {len(cids)} community lớn nhất")

    reports = []
    for cid in tqdm(cids, desc="Community reports"):
        try:
            rep = summarize_community(cid)
        except Exception as e:
            print(f"  community {cid} lỗi: {type(e).__name__}")
            continue
        if rep:
            reports.append(rep)
    return pd.DataFrame(reports)


community_reports_df = build_community_reports(community_df)
display(community_reports_df)

# --- Bước 5: Global Search — retrieve trên REPORT (không phải chunk) ---
global_report_index = None
global_report_texts = []

def build_global_report_index(reports_df):
    """Index FAISS trên summary của community -> trả lời câu hỏi ở tầng 'high-level'."""
    global global_report_index, global_report_texts
    if reports_df.empty:
        print("Không có report -> bỏ qua global search.")
        return
    global_report_texts = [
        f"[community {r.community_id}] {r.title}: {r.summary}"
        for r in reports_df.itertuples(index=False)
    ]
    vecs = get_embedder().encode(
        global_report_texts, batch_size=32,
        show_progress_bar=False, normalize_embeddings=True,
    ).astype("float32")
    global_report_index = faiss.IndexFlatIP(vecs.shape[1])
    global_report_index.add(vecs)
    print(f"✅ Global report index: {len(global_report_texts)} report")


def global_search(question, k=3):
    if global_report_index is None:
        return {"answer": "", "context": "", "route": "NO_REPORT"}
    q = get_embedder().encode(
        [question], normalize_embeddings=True
    ).astype("float32")
    _, idxs = global_report_index.search(q, min(k, len(global_report_texts)))
    context = "\\n\\n".join(global_report_texts[i] for i in idxs[0] if i >= 0)
    out = generate_answer(question, context)
    out.update({"context": context, "route": "GLOBAL_COMMUNITY_REPORT"})
    return out


build_global_report_index(community_reports_df)

# Demo: câu hỏi "toàn cục" kiểu này Flat RAG rất yếu vì không có chunk nào chứa cả bức tranh.
_global_demo_q = "What are the main themes of acquisitions and partnerships across these tech companies?"
_g = global_search(_global_demo_q)
print(f"\\nQ: {_global_demo_q}\\nroute={_g['route']}\\n\\n{_g['answer'][:1200]}")
'''

# ---------------------------------------------------------------------------
# C. Bonus C — driver cho self-correction (cell gốc chỉ định nghĩa hàm)
# ---------------------------------------------------------------------------

CELL_SELF_CORRECTION_DEMO = '''# @bonus:self_correction_demo
#@title Bonus C (driver) — Chạy self-correction trên câu multi-hop thật
# Cell trên chỉ ĐỊNH NGHĨA self_correcting_context() mà không gọi -> không có output để chấm.
# Ở đây chạy thật và ghi lại route đã chọn cho từng câu.
#
# Stop condition (bắt buộc theo Bonus C): tối đa hop 2 -> hop 3 -> vector fallback,
# tức nhiều nhất 3 lần retrieve + 2 lần LLM kiểm tra sufficiency. Không có vòng lặp mở.
SELF_CORRECTION_DEMO_N = 5

_demo_pool = golden_df[golden_df.group.isin(["multi-hop", "cross-doc"])]
_demo_qs = (_demo_pool if len(_demo_pool) else golden_df).head(SELF_CORRECTION_DEMO_N)

_rows = []
for q in tqdm(_demo_qs.itertuples(index=False), total=len(_demo_qs), desc="Self-correction"):
    try:
        r = self_correcting_context(q.question)
    except Exception as e:
        _rows.append({"id": q.id, "group": q.group, "route": f"ERROR:{type(e).__name__}",
                      "context_chars": 0, "missing": ""})
        continue
    _rows.append({
        "id": q.id,
        "group": q.group,
        "route": r["route"],
        "context_chars": len(r["context"]),
        "missing": r["missing"][:180],
    })

self_correction_df = pd.DataFrame(_rows)
display(self_correction_df)
print("Phân bố route:", self_correction_df.route.value_counts().to_dict())
print(
    "Stop condition: hop2 -> hop3 -> vector fallback, tối đa 3 lần retrieve "
    "+ 2 lần LLM sufficiency check; không có vòng lặp không giới hạn."
)
'''


def main() -> int:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    before = len(cells)

    # Bảo đảm patch_notebook.py đã chạy (nếu chưa, index/nội dung sẽ khác hẳn).
    if not any("GOLDEN_EVIDENCE_ROWS" in src(c) for c in cells):
        raise SystemExit(
            "[FAIL] Chưa thấy GOLDEN_EVIDENCE_ROWS -> hãy chạy tools/patch_notebook.py trước."
        )

    insert_after(
        cells,
        anchor="Challenge A — Near Dedup",
        marker="# @bonus:near_dedup",
        text=CELL_NEAR_DEDUP,
        label="A. near-dedup MinHash+LSH",
    )
    insert_after(
        cells,
        anchor="community_df = build_communities()",
        marker="# @bonus:community_reports",
        text=CELL_COMMUNITY_REPORTS,
        label="B. community report + global search",
    )
    insert_after(
        cells,
        anchor="def self_correcting_context",
        marker="# @bonus:self_correction_demo",
        text=CELL_SELF_CORRECTION_DEMO,
        label="C. self-correction driver",
    )

    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print("THÊM CELL BONUS XONG:")
    for line in CHANGELOG:
        print(line)
    print(f"\nSố cell: {before} -> {len(cells)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
