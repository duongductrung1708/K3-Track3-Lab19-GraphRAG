# Lab 19 — GraphRAG vs Flat RAG · Phân tích yêu cầu & Checklist thực thi

> Nguồn: `ASSIGNMENT.md`, `RUBRIC.md`, `README.md`, `Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb` (37 cells).

## 0. Bản chất bài lab

Notebook **không phải bài code-from-scratch**. Toàn bộ 37 cell đã có code đầy đủ (0 `TODO`, 0 `NotImplementedError`).
Điểm mấu chốt: **mọi lệnh gọi thực thi đều đang bị comment**, và **cả 37 cell đều có `outputs = 0`** (chưa từng chạy).

```
Cell 007: # connect_neo4j() / # setup_graph_schema()
Cell 008: # raw_df = load_news(DATA_PATH) ...
Cell 012: # coref_df = run_coref(extraction_source)
Cell 014: # raw_triples_df, extraction_errors_df = run_extraction(...)
Cell 016: # entity_map, entity_resolution_audit_df = build_resolution_map(...)
Cell 017: # bulk_insert_nodes(nodes_df) / # bulk_insert_edges(triples_df)
Cell 018: # graph_counts, top_degree_df = graph_checks()
Cell 020: # build_flat_index(chunks_df)
Cell 022: # build_entity_matcher(nodes_df)
Cell 028: # eval_results_df = run_evaluation(golden_df)
Cell 029: # comparison_df / # to_csv(...)
Cell 031: # test_supernode_policy() / # show_resolution_audit(...)
Cell 034: # community_df = build_communities()
```

=> **Việc phải làm = uncomment + chạy thật + sinh dữ liệu thật + viết báo cáo dựa trên số liệu thật.**
RUBRIC yêu cầu `Restart & Run All` thành công và "notebook đã chạy đầy đủ output các cell" — nếu để nguyên comment thì Run All gần như không sinh output nào.

---

## 1. Deliverables bắt buộc (theo RUBRIC 100đ + 10 bonus)

| # | File | Trọng số | Trạng thái |
|---|------|----------|-----------|
| D1 | `Day19_..._Lab_Guide.ipynb` đã chạy đủ output | 40đ (Implementation) | 🚧 code sẵn sàng, đã chạy thật 2/22 code cell (chặn ở cell 1.3 vì thiếu data) |
| D2 | `outputs/graphrag_eval_results.csv` | 20đ (Eval) | ❌ chưa có (cần P5) |
| D3 | `outputs/graphrag_vs_flatrag_summary.csv` | 20đ (Eval) | ❌ chưa có (cần P5) |
| D4 | `reports/lab_report.md` (điền đủ 2 phần) | 20đ (Thuyết minh) | ❌ còn nguyên template (chờ số liệu thật) |
| D5 | `data/golden_dataset.csv` (5 câu, đủ `reference_answer`) | 6đ (3.1) | ✅ **XONG** — 50 câu, vượt yêu cầu (`data/graphrag_golden_50_first5000.csv`) |
| D6 | Bonus: Community Detection / Self-Correction / Near-Dedup | +10đ | ❌ code scaffold có, chưa chạy |

### Tiêu chí "phải chứng minh bằng output" (Failure Modes — 20đ)
- [ ] **2.1** Super-node: node `degree > 100` → cắt còn `≤ 50` cạnh mới nhất; `GLOBAL_EDGE_CAP = 250` → chạy `test_supernode_policy()`
- [ ] **2.2** Provenance: Cypher check `invalid_provenance_edges == 0` (thiếu → **-5đ**)
- [ ] **2.3** `entity_resolution_audit_df` ≥ **10 dòng**, có đủ 3 nhãn `MERGE_MANUAL` / `MERGE_VECTOR` / `REJECT_GUARD`

### Ràng buộc kỹ thuật cứng
- [ ] Bulk insert **bắt buộc** `UNWIND $rows AS row` batch 1000 — cấm `MERGE`/`CREATE` từng row
- [ ] Constraint `(n:Entity).id` UNIQUE + index `name_norm`
- [ ] Mọi edge có `source_chunk_id`, `published_date`, `evidence`, `confidence`
- [ ] Scale guard: `LAB_MAX_ARTICLES=1500`, `LAB_MAX_CHUNKS=3000`, `EXTRACTION_MAX_CHUNKS=400`, `CHUNK_WORDS=220/40`
- [ ] Golden Dataset đủ 3 nhóm `factoid` / `multi-hop` / `cross-doc`
- [ ] Judge 3 thang điểm 1–5: Comprehensiveness / Faithfulness / Multi-hop reasoning + `rationale`
- [ ] **Không hard-code API key / password** vào notebook (vi phạm **-10đ**)

---

## 2. Lỗi & mâu thuẫn phát hiện trong repo (phải xử lý)

| ID | Vấn đề | Ảnh hưởng |
|----|--------|-----------|
| **B1** | `.gitignore` có `*.csv` + ngoại lệ `!reports/*.csv`, nhưng deliverable CSV lại nằm ở **`outputs/`** → 2 file nộp bị git bỏ qua, push lên GitHub sẽ **mất** | **-5đ** ("không xuất được 2 file CSV") |
| **B2** | Notebook hard-code path Colab: `/content/hackernoon_subset.csv`, `/content/golden_dataset.csv`, `/content/graphrag_eval_checkpoint.csv`, và `to_csv("/content/...")` → chạy local Windows là crash; CSV không rơi vào `outputs/` | D2, D3 |
| **B3** | Thiếu thư mục `data/` + `data/golden_dataset.csv` dù README mô tả có | D5, 6đ |
| **B4** | `golden_dataset.csv` bị `.gitignore` chặn 2 lớp (`*.csv` và dòng riêng `golden_dataset.csv`) | D5 |
| **B5** | G02–G05 có `reference_answer` rỗng; `validate_golden(require_answers=True)` sẽ **raise** → chặn cả pipeline eval | D2, D3 |
| **B6** | `ASSIGNMENT.md` bị lỗi file: câu 170 đứt giữa dòng, khối "Deliverables/Checklist" bị lặp 2 lần, có mojibake `��` (dòng 217) | Gây mâu thuẫn số file báo cáo |
| **B7** | Mâu thuẫn báo cáo: README + template → **1 file** `reports/lab_report.md`; RUBRIC 4.1–4.3 + nửa sau ASSIGNMENT → **3 file** `technical_defense.md`, `failure_analysis.md`, `reflection_[HọTên].md` (thiếu file = -5đ/file) | **Xử lý: làm cả 2** — `lab_report.md` đầy đủ + 3 file split trỏ về nó |
| **B8** | Mâu thuẫn thư mục CSV: RUBRIC 3.3 ghi `reports/`, README + ASSIGNMENT ghi `outputs/` | **Xử lý: ghi cả 2 chỗ** |
| **B9** | Cell 003 cài thêm `spacy`, `langchain-community`, `llama-index` — không dùng trong code, không có trong `requirements.txt`, tốn vài phút cài | Chậm, dễ conflict |
| **B10** | Không có cơ chế đọc `.env` (chỉ `google.colab.userdata` → fallback `os.environ`) → local phải tự export biến môi trường | Setup local |

---

## 3. Điều kiện tiên quyết còn thiếu

| Dependency | Bắt buộc | Trạng thái máy hiện tại |
|-----------|----------|------------------------|
| Neo4j 5.x / AuraDB | ✅ | ❌ chưa có (Docker 29.6.2 **đã có** → chạy container được) |
| `GROQ_API_KEY` + `GROQ_MODEL` | ✅ (coref, NER+RE, seed, generator) | ❌ chưa có `.env` |
| `HF_TOKEN` | ✅ (stream dataset) | ❌ |
| `OPENAI_API_KEY` (judge) | ⚠️ thay được bằng Groq | `openai 2.50.0` đã cài, chưa rõ key |
| `sentence-transformers`, `faiss-cpu`, `torch`, `neo4j`, `datasets`, `groq` | ✅ | ❌ chưa cài; **Python local là 3.14.6** → rủi ro không có wheel cho torch/faiss |

---

## 4. Kế hoạch thực thi

- [x] **P0** Chốt môi trường chạy (Colab vs Local+Docker) & thu thập API keys — ✅ **XONG 2026-08-19** (chi tiết §5)
- [x] **P1** Sửa hạ tầng repo: `.gitignore` (B1, B4), tạo `data/golden_dataset.csv` (B3), path portable + `.env` loader (B2, B10), gọn install cell (B9) — ✅ **XONG** (§5 P1)
- [x] **P2** Uncomment toàn bộ driver call → notebook `Restart & Run All` chạy được thật — ✅ **XONG** (§5 P2)
- [ ] **P3** Chạy M1→M2: dedup, chunk, coref, NER+RE, entity resolution, UNWIND ingest, sanity check `invalid_provenance_edges == 0` — 🚧 **CHẶN bởi B11**
- [ ] **P4** Chạy M4: Flat FAISS index, seed match, BFS + super-node cap, hybrid answer
- [ ] **P5** Điền `reference_answer` thật cho G02–G05 **từ chính graph/dữ liệu đã nạp** (B5), rồi chạy LLM-as-a-Judge → xuất D2, D3
- [ ] **P6** Chạy `test_supernode_policy()` + `show_resolution_audit()`, thu số liệu top-degree & cặp `REJECT_GUARD` similarity > 0.85
- [ ] **P7** Viết `reports/lab_report.md` bằng **số liệu thật** + 3 file split (B7)
- [ ] **P8** Bonus: NetworkX community detection + self-correction retrieval + near-dedup (MinHash/LSH, cấm O(N²))
- [ ] **P9** Rà submission checklist, commit & push

---

## 5. NHẬT KÝ THỰC THI (cập nhật khi xong từng bước)

### ✅ P0 — Chốt môi trường & API keys (xong 2026-08-19)

**Quyết định: chạy LOCAL + Docker Neo4j** (không dùng Colab, không dùng AuraDB).
Lý do: `.venv` đã tạo sẵn ở local, Docker 29.6.2 có sẵn → tự chủ hoàn toàn, không cần đăng ký AuraDB.

| Hạng mục | Kết quả | Ghi chú |
|---|---|---|
| Python | ✅ 3.14.6 | Rủi ro trong §3 **không xảy ra** — có đủ wheel cp314 |
| `.venv` deps | ✅ cài xong (`PIP_EXIT=0`) | torch `2.13.0+cpu`, faiss `1.15.0`, sentence-transformers `6.0.0`, pandas `3.0.5`, neo4j driver `6.2.0`, groq `1.6.0`, openai `3.3.0`, datasets `5.0.1`, networkx `3.6.1` |
| Neo4j | ✅ **5.26.29** chạy container `lab19-neo4j` | `docker run -d --name lab19-neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/<gen> -e NEO4J_server_memory_heap_max__size=2G -e NEO4J_server_memory_pagecache_size=1G neo4j:5.26` |
| Neo4j driver API | ✅ verify | `verify_connectivity()`, `session.run(q,**params)`, `r.data()`, `consume()` đều OK trên driver 6.x |
| `GROQ_API_KEY` | ✅ hợp lệ | |
| `OPENAI_API_KEY` | ❌ **429 insufficient_quota** | key hợp lệ nhưng hết quota → **không dùng làm judge** |
| `HF_TOKEN` | ✅ hợp lệ (user `Koonee`, fineGrained) | nhưng xem **B11** bên dưới |

#### Sự cố đã xử lý trong P0
- **Docker daemon chưa chạy** + `Docker Desktop.exe` **không** ở `C:\Program Files\Docker\` mà ở `%LOCALAPPDATA%\Programs\DockerDesktop\` → đã start bằng đường dẫn đúng.
- **`pip install` fail `WinError 32`**: còn `pip.exe` (PID 20584) + `python.exe` (PID 24868) treo từ session trước giữ lock `.venv\Lib\site-packages\pydantic\`. Đã kill 2 process → cài lại thành công.
  - Bài học: `pip ... | tail` **che mất** exit code (báo `exited with code 0` dù ERROR) → phải `> log 2>&1; echo $?`.

#### B12 (mới) — `GROQ_MODEL` trong `.env.example` không tồn tại
`llama-3.3-70b-versatile` → **404 model_not_found**. Key chỉ có 13 model; loại chat dùng được:
`openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.6-27b`, `groq/compound{,-mini}`.

**Đã chốt (cả 3 đã test pass JSON mode):**
- `GROQ_MODEL=openai/gpt-oss-120b` — coref, NER+RE, seed, generator (chất lượng cao nhất)
- `JUDGE_PROVIDER=groq` + `JUDGE_MODEL=qwen/qwen3.6-27b` — judge **khác họ model** với generator để giảm
  self-preference bias (thay cho OpenAI đã hết quota). → **phải ghi rõ trong `lab_report.md`**.

#### 🚧 B11 (mới) — BLOCKER: dataset HackerNoon bị GATED
```
DatasetNotFoundError: Dataset 'HackerNoon/tech-company-news-data-dump' is a gated dataset
on the Hub. Visit the dataset page ... to ask for access.
```
`HF_TOKEN` hợp lệ nhưng account chưa được cấp quyền. **Chặn P3→P8** (mọi thứ downstream cần data).
Cần user thao tác tay:
1. Mở https://huggingface.co/datasets/HackerNoon/tech-company-news-data-dump → **Agree / Request access**
2. Token `fineGrained` phải bật scope **"Read access to contents of all public gated repos you can access"**
   (hoặc thay bằng token `Read` thường).

---

### ✅ P1 — Hạ tầng repo (xong 2026-08-19)

Toàn bộ patch notebook được đóng gói trong `tools/patch_notebook.py` (idempotent, có assert anchor —
anchor không khớp thì **raise** chứ không patch im lặng sai chỗ). Chạy lại an toàn nhiều lần.

| Bug | Cách xử lý | Nơi kiểm chứng |
|---|---|---|
| B1, B4 | Bỏ hẳn `*.csv` + `golden_dataset.csv` khỏi `.gitignore`; chỉ ignore `data/hackernoon_subset.csv` (sinh lại được) và `outputs/graphrag_eval_checkpoint.csv` (file tạm) | `.gitignore` |
| B2, B10 | `_find_project_root()` + `DATA_DIR/OUTPUTS_DIR/REPORTS_DIR`; `get_secret()` theo thứ tự Colab userdata → `.env` → `os.environ`. Hết `/content/`, không hard-code key | cell 004 |
| B3, B5 | `data/graphrag_golden_50_first5000.csv` — **50 câu** (23 multi-hop / 22 cross-doc / 5 factoid), `reference_answer` + `reference_evidence` đã điền đủ. Bản `_detailed.csv` thêm 15 cột (`evidence_row_ids_0based`, `expected_hops`, `seed_entities`, `gold_reasoning`, …) | `data/` |
| B8 | Cell 029 ghi CSV vào **cả** `outputs/` và `reports/` | cell 029 |
| B9 | Cell 003 bỏ `spacy`/`langchain-community`/`llama-index`, chỉ cài package thực sự thiếu | cell 003 |

**Sửa thêm ngoài checklist gốc — evidence-aware sampling (quan trọng):**
Golden dataset trải trên row 33..4997 nhưng `LAB_MAX_ARTICLES=1500` và `EXTRACTION_MAX_CHUNKS=400`.
Bản gốc sample ngẫu nhiên + `chunks_df.head(400)` → chỉ ~19/51 article evidence trụ lại, phần lớn câu
hỏi thành *không thể trả lời* và phép so sánh GraphRAG vs Flat RAG mất ý nghĩa.
→ `standardize_news()` giữ **trọn** article evidence rồi lấp distractor theo `SEED`;
`select_extraction_source()` ưu tiên chunk của article evidence. Vẫn tôn trọng scale guard.

### ✅ P2 — Uncomment driver call (xong 2026-08-19)

Đã quét lại toàn notebook: **0 dòng driver-call còn bị comment** (chỉ còn comment giải thích).
Cell 028 được nâng cấp thêm `resume=True` — 50 câu × ~4 lệnh gọi LLM ≈ 200 call, mất mạng/rate limit
giữa đường thì chạy lại bỏ qua câu đã xong (đọc `outputs/graphrag_eval_checkpoint.csv`).

### ✅ Công cụ chạy headless (mới, 2026-08-19)

| Script | Việc |
|---|---|
| `tools/run_notebook.py` | "Restart & Run All" trong terminal, **ghi output trực tiếp vào .ipynb và save sau MỖI cell** → mất mạng giữa pipeline vẫn giữ output đã có. Cờ: `--dry-run`, `--start-from N`, `--stop-after N`, `--timeout`, `--allow-errors` |
| `tools/check_hf_access.py` | Tách rõ 3 nguyên nhân của lỗi gated (token chết / chưa Agree / token thiếu scope) — `load_dataset()` chỉ báo chung chung. Có `--wait` để poll |

Đã cài `nbclient`, `nbformat`, `ipykernel`, `ipywidgets` vào `.venv` và đăng ký kernelspec `python3`.
**Đã chạy thật cell 003 + 004:** dependencies đủ (không cài gì), `PROJECT_ROOT` trỏ đúng repo,
7/7 secret nhận diện được. Neo4j `5.26.29` verify OK, database đang **rỗng 0 node** (sẵn sàng ingest).

### 🚧 Trạng thái hiện tại — chỉ còn chờ quyền HF

Chẩn đoán lại B11 chính xác hơn (bằng `tools/check_hf_access.py`):
- `whoami` OK, account `Koonee`, token **còn sống**.
- Metadata dataset đọc được (`gated=auto`), README `resolve` trả **200**.
- Nhưng data file `cleanedCompanyNews.csv` trả **403 "you are not in the authorized list"**
  → **account chưa bấm Agree**, và token `fineGrained` có `global_scopes=[]` tức **thiếu scope gated**.

Đã loại trừ các đường vòng:
- Mirror `pacozaa/tech-company-news-data-dump-clean` (không gated): chỉ **2923 dòng / 2 cột**
  (`companyName`, `description`) — không có `title`/`date`/`url` → **không dùng được**.
- HF hub cache chỉ còn `README.md` (streaming không cache data).
- `data/hackernoon_subset.csv` **không** có trong git history (bị ignore) và **không** có trong
  Recycle Bin (thứ bị xoá hôm nay là các folder lab khác).

**Việc user cần làm (≈1 phút):** xem hướng dẫn in ra bởi `tools/check_hf_access.py`.
Xong thì cell 1.3 tự stream lại đúng 5000 dòng đầu → row index khớp golden dataset tuyệt đối.
