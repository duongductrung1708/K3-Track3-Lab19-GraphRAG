"""Cập nhật các markdown cell cho khớp bản patch (tránh hướng dẫn lỗi thời)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb"

CELL_002 = """# PHẦN 1 — SETUP & PREPROCESSING

### Secrets — Colab Secrets HOẶC file `.env` ở gốc repo

Notebook đọc secret theo thứ tự: **Colab `userdata` → `.env` → biến môi trường**.
Chạy local thì `cp .env.example .env` rồi điền:

- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`
- `GROQ_API_KEY`, `GROQ_MODEL` — dùng cho coref, NER+RE, seed extraction, generator
- `JUDGE_PROVIDER`, `JUDGE_MODEL` (+ `OPENAI_API_KEY` nếu judge là OpenAI)
- `HF_TOKEN` để stream dataset từ Hugging Face

`.env` đã nằm trong `.gitignore`. **Không hard-code API key vào notebook nộp bài.**

### Neo4j local bằng Docker (thay cho AuraDB)

```bash
docker run -d --name lab19-neo4j -p 7474:7474 -p 7687:7687 \\
  -e NEO4J_AUTH=neo4j/<password-cua-ban> \\
  -e NEO4J_server_memory_heap_max__size=2G \\
  -e NEO4J_server_memory_pagecache_size=1G \\
  neo4j:5.26
```

Rồi đặt `NEO4J_URI=bolt://localhost:7687` trong `.env`.
"""

CELL_005 = """## 1.3 — Download HackerNoon Dataset bằng Hugging Face Streaming

Cell dưới stream dataset **`HackerNoon/tech-company-news-data-dump`** và ghi dần ra CSV,
nên không cần nạp toàn bộ dataset vào RAM.

### Giới hạn đang dùng

- `LIMIT_ROWS = GOLDEN_SOURCE_ROWS = 5000` — **chốt cứng 5000 dòng đầu**.
- `PRIORITIZE_MB = False` — dừng theo **số dòng**, không theo dung lượng.
- `LIMIT_MB = 300` — chỉ còn là chặn an toàn.

> **Vì sao đúng 5000 dòng?** Golden dataset (`data/graphrag_golden_50_first5000*.csv`)
> được soạn trên đúng 5000 dòng đầu của file này, cột `evidence_row_ids_0based` trỏ tới
> row index 0-based trong đó. Cell này ghi **tuần tự theo thứ tự stream**, nên tải lại
> 5000 dòng đầu sẽ tái lập chính xác cùng tập article — evidence vẫn khớp.

### Lưu ý

- `HF_TOKEN` đọc từ `.env` (local) hoặc Colab Secrets — không hard-code.
- Dataset là **gated**: phải mở trang dataset trên Hugging Face và hoàn tất
  **Agree / Request access**; token dạng `fineGrained` cần bật scope
  *"Read access to contents of all public gated repos you can access"*.
- Cell **idempotent**: nếu `DATA_PATH` đã có đủ 5000 dòng thì bỏ qua download.
  Muốn tải lại, đặt `FORCE_REDOWNLOAD = True`.
- Đích ghi là `DATA_PATH` (mặc định `data/hackernoon_subset.csv` trong repo),
  không còn `/content/...` như bản gốc.
"""

CELL_001 = """## ⏳ Timeline

| Phút | Nội dung |
|---|---|
| 00–15 | Setup, load, dedup, chunk, coreference |
| 15–45 | NER/RE, entity resolution, Neo4j bulk insert |
| 45–75 | Flat RAG, graph traversal, hybrid retrieval |
| 75–105 | Golden Dataset, LLM-as-a-Judge, comparison |
| 105–120 | Failure-mode tests, bonus, export, thuyết minh |

### Scale guard
Trong lab 2 giờ, không nên gửi toàn bộ 350MB qua LLM. Mặc định dùng subset:
- `LAB_MAX_ARTICLES = 1500`
- `LAB_MAX_CHUNKS = 3000`
- `EXTRACTION_MAX_CHUNKS = 400`

Kiến trúc phải scale được; volume trong giờ lab chỉ dùng để chứng minh pipeline.

### Cách chọn subset (khác bản gốc — có chủ đích)

Bản gốc chọn `df.sample(1500, random_state=SEED)` ngẫu nhiên và
`chunks_df.head(400)` theo thứ tự file. Với golden dataset trải trên row 33..4997,
cách đó chỉ giữ **19/51** article evidence → phần lớn câu hỏi eval thành
không-thể-trả-lời và phép so sánh Flat vs Graph mất ý nghĩa.

Bản này giữ **nguyên số** của scale guard nhưng chọn mẫu **deterministic + evidence-aware**:

1. `standardize_news()` — giữ trọn article evidence của golden dataset, lấp phần còn lại
   tới đủ 1500 bằng shuffle theo `SEED` (làm distractor cho retrieval).
2. `select_extraction_source()` — trong budget 400 chunk, nạp chunk của article evidence
   trước, còn chỗ mới lấp bằng chunk đầu file.

Hệ quả cần biết khi đọc kết quả: **Flat RAG index phủ toàn bộ ~3000 chunk**, còn
**graph chỉ phủ 400 chunk đã extract**. Tức Flat RAG được truy cập tập văn bản rộng hơn
(superset) — cách đọc này *bất lợi* cho GraphRAG, nên chênh lệch nghiêng về GraphRAG
không phải do graph được ưu ái về dữ liệu.
"""

REPLACEMENTS = {
    1: ("Scale guard", CELL_001),
    2: ("Secrets", CELL_002),
    5: ("Streaming", CELL_005),
}


def main() -> int:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    changed = 0
    for idx, (anchor, text) in REPLACEMENTS.items():
        cell = cells[idx]
        if cell["cell_type"] != "markdown":
            raise SystemExit(f"[FAIL] cell {idx} không phải markdown")
        body = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        if body.strip() == text.strip():
            print(f"  = cell {idx:03d} (đã cập nhật trước đó)")
            continue
        if anchor not in body:
            raise SystemExit(f"[FAIL] cell {idx}: không thấy anchor {anchor!r}")
        cell["source"] = text.splitlines(keepends=True)
        print(f"  * cell {idx:03d} markdown -> cập nhật")
        changed += 1

    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Xong: {changed} markdown cell cập nhật.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
