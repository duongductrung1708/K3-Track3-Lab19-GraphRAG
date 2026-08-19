# Báo Cáo Thực Hành & Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** [Họ và Tên]  
**Khóa học:** AICB-K34 · Track 3: GraphRAG  
**Ngày thực hiện:** [Ngày/Tháng/Năm]

---

## 📌 PHẦN 1: THUYẾT MINH KỸ THUẬT & PHÂN TÍCH CA LỖI

### 1. Coreference Resolution (Phân giải đại từ)

> **Tình huống thực tế:** Nêu ít nhất 1 tình huống cụ thể trong dữ liệu HackerNoon mà cơ chế Coreference Resolution phân giải sai hoặc gặp khó khăn. Hậu quả của nó đối với Knowledge Graph là gì?

_Trả lời:_

- **Ví dụ từ dữ liệu:** [Trích dẫn chunk_id hoặc câu văn cụ thể]
- **Hiện tượng:** [Ví dụ: 'The company' bị nhầm sang công ty được nhắc đến ở câu trước thay vì chủ ngữ chính]
- **Hậu quả đối với Graph:** [Ví dụ: Tạo ra False Edge gán nhầm sự kiện M&A cho đối thủ cạnh tranh]

---

### 2. Entity Resolution Threshold & Lexical Guard

> **Ngưỡng & Cơ chế Guard:** Bạn chọn ngưỡng cosine similarity là bao nhiêu cho vector matching? Trích dẫn 1 cặp thực thể có độ tương đồng vector cao ($> 0.85$) nhưng bị Lexical Guard chặn không cho gộp (Reject) và giải thích lý do.

_Trả lời:_

- **Ngưỡng cosine similarity:** `threshold = ...` (ví dụ: 0.90)
- **Cặp thực thể bị Guard chặn:** `[Thực thể A]` vs `[Thực thể B]` (Ví dụ: `Sam Altman` vs `Steve Altman` hoặc `Apple` vs `Apple Music`)
- **Lý do chặn:** [Lý do ngữ nghĩa tại sao không được gộp 2 thực thể này]

---

### 3. Đồ thị & Super-node Mitigation

> **Đặc trưng đồ thị & Cắt tỉa cạnh:** Top 3 thực thể có bậc (degree) cao nhất trong đồ thị là gì? Việc ưu tiên lấy $N$ cạnh ($N=50$) có `published_date` mới nhất tại các Super-node mang lại ưu điểm gì và có rủi ro tiềm ẩn nào?

_Trả lời:_

- **Top 3 Super-nodes:**

| Hạng | Tên thực thể | Loại thực thể (Type) | Bậc kết nối (Degree) |
| ---- | ------------ | -------------------- | -------------------- |
| 1    |              |                      |                      |
| 2    |              |                      |                      |
| 3    |              |                      |                      |

- **Ưu điểm & Rủi ro của Temporal Mitigation:**
  - _Ưu điểm:_ [Giảm thiểu bùng nổ context, giữ lại thông tin cập nhật nhất...]
  - _Rủi ro:_ [Nếu câu hỏi liên quan đến sự kiện lịch sử trong quá khứ xa có thể bị cắt mất...]

---

### 4. So sánh Thực nghiệm (Flat RAG vs GraphRAG)

#### Bảng tổng hợp Benchmark (LLM-as-a-Judge):

| Tiêu chí đánh giá             | Flat RAG | GraphRAG | Độ chênh lệch ($\Delta$) | Nhận xét phân tích |
| ----------------------------- | -------- | -------- | ------------------------ | ------------------ |
| **Comprehensiveness (1–5)**   |          |          |                          |                    |
| **Faithfulness (1–5)**        |          |          |                          |                    |
| **Multi-hop Reasoning (1–5)** |          |          |                          |                    |
| **Latency trung bình (s)**    |          |          |                          |                    |
| **Token usage trung bình**    |          |          |                          |                    |

#### Phân tích 2 Ca lỗi Điển hình:

1. **Ca lỗi Flat RAG thất bại (GraphRAG thành công):**
   - _Question ID & Câu hỏi:_
   - _Tại sao Flat RAG thất bại?_ [Ví dụ: Vector search không kết nối được 2 chunks chứa thông tin rời rạc...]
   - _GraphRAG đã giải quyết như thế nào?_ [Ví dụ: Graph traversal qua cạnh A -> B -> C...]
2. **Ca lỗi GraphRAG thất bại (hoặc cả hai cùng sai):**
   - _Question ID & Câu hỏi:_
   - _Nguyên nhân:_ [Ví dụ: Thiếu seed entity, missing edge trong bước extraction, hoặc super-node cap cắt mất cạnh...]
   - _Đề xuất khắc phục:_ [...]

---

### 5. Đánh đổi (Trade-offs) & Kiểm soát AI Coding Agent

> **Trade-offs, Agent Control & Scale 350MB:**
>
> - So sánh sự đánh đổi giữa GraphRAG vs Flat RAG về Latency, Token và Indexing Overhead.
> - Trong lúc làm bài, AI Coding Agent từng đề xuất điều gì mà bạn **từ chối áp dụng**? Tại sao?
> - Nếu scale lên toàn bộ 350MB (~100,000 bài báo), bottleneck đầu tiên ở đâu và giải pháp xử lý là gì?

_Trả lời:_

- **Đánh đổi Quality vs Cost vs Latency:** [...]
- **Quyết định từ chối AI Coding Agent:** [Ví dụ: Từ chối thuật toán $O(N^2)$ pairwise cosine trên toàn bộ dataset vì gây tràn RAM/OOM...]
- **Giải pháp scale 350MB:** [Ví dụ: Async batch extraction với worker queue, HNSW index với blocking cho Entity Resolution, Community Partitioning...]

---

## 📌 PHẦN 2: SUY NGẪM & KẾ HOẠCH ĐỒ ÁN (Reflection & Action Plan)

### 1. Mapping Bài giảng vào Code

| Khái niệm trong bài giảng          | Module tương ứng | Hàm / Khối code cụ thể                       | Quan sát thực tế & Đánh giá |
| ---------------------------------- | ---------------- | -------------------------------------------- | --------------------------- |
| **Conservative Coreference**       | Module 1         | `resolve_coref_batch()`                      | ...                         |
| **Schema & Allowlist Guard**       | Module 2         | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS`    | ...                         |
| **Bulk Cypher Ingestion**          | Module 2         | `bulk_insert_nodes()`, `bulk_insert_edges()` | ...                         |
| **Entity Resolution & Union-Find** | Module 3         | `build_resolution_map()`, `UF`               | ...                         |
| **Super-node Degree Cap**          | Module 4         | `retrieve_graph_context()`                   | ...                         |
| **LLM-as-a-Judge Evaluation**      | Module 5         | `judge_answer()`                             | ...                         |

---

### 2. Quá trình Debugging & Bài học

- **Lỗi kỹ thuật phức tạp nhất gặp phải:** [...]
- **Cách bạn đã xử lý thành công:** [...]

---

### 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)

- **Tên đồ án / Dự án:** [Tên dự án]
- **Đặc thù bài toán & Lý do chọn giải pháp:** [Tại sao bài toán của bạn cần GraphRAG hay chỉ cần Flat/Hybrid RAG?]
- **Cấu trúc Node & Relation dự kiến:**
  - Nodes: `...`
  - Relations: `...`
- **Chiến lược xử lý Super-node & Entity Resolution:** [...]

---

## 🎯 TỰ ĐÁNH GIÁ

| Tiêu chí                             | Điểm tự chấm (1–5) | Ghi chú |
| ------------------------------------ | ------------------ | ------- |
| Mức độ hiểu bài giảng GraphRAG       |                    |         |
| Khả năng kiểm soát AI Coding Agent   |                    |         |
| Chất lượng đồ thị tri thức xây dựng  |                    |         |
| Khả năng phân tích và debug hệ thống |                    |         |
