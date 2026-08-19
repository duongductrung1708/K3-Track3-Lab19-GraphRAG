"""Sinh corpus + golden dataset GIẢ LẬP để pre-flight toàn pipeline P3->P8.

Vì sao cần: dataset thật (HackerNoon) đang bị gated (B11) nên chưa chạy được P3.
Nhưng một lần chạy thật tốn ~30-60 phút và ~200 lệnh gọi LLM — nếu để crash giữa
đường mới phát hiện bug thì rất đắt. Script này tạo corpus nhỏ (~130 article) để
chạy end-to-end thật (Neo4j thật + Groq thật) trong vài phút, nhằm bắt hết bug
trước khi có quyền dataset thật.

Corpus = 2 phần:
  1. FIXTURE (16 article tự soạn): cố ý dựng sẵn multi-hop chain, cross-doc
     temporal shift, hub node, và các cặp tên gần giống để kích hoạt đủ 3 nhãn
     entity-resolution (MERGE_MANUAL / MERGE_VECTOR / REJECT_GUARD).
  2. REAL TEXT (~114 article) lấy từ mirror KHÔNG gated
     `pacozaa/tech-company-news-data-dump-clean` để văn bản có độ nhiễu thật.

Tên cột đặt theo camelCase (`description`, `articleTitle`, `publishedDate`,
`sourceUrl`) — KHÁC hoàn toàn danh sách candidate gốc trong notebook. Đây là chủ
ý: nếu `standardize_news()` không nhận diện được cột thì pre-flight phải fail ở
đây, chứ không phải fail sau khi user đã chờ HF cấp quyền.

Chạy:  .venv/Scripts/python.exe tools/make_preflight_data.py
Output: .preflight/corpus.csv, .preflight/golden.csv, .preflight/golden_detailed.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".preflight"
MIRROR = OUT / "mirror.parquet"

# ---------------------------------------------------------------------------
# 1. FIXTURE — dựng chủ đích để mọi nhánh code đều chạy
# ---------------------------------------------------------------------------
# Đồ thị mong đợi:
#   Veritas Capital -INVESTED_IN-> Aurelia Systems -ACQUIRED-> Nimbus Robotics
#                                                              -DEVELOPED-> PathSense
#   Mira Sandoval -WORKED_AT-> Aurelia Systems ; -FOUNDED-> Halcyon Labs
#   Veritas Capital -INVESTED_IN-> Halcyon Labs        (multi-hop 2 chặng)
#   Orion Cloud -PARTNERED_WITH-> Aurelia (2022) rồi -ACQUIRED-> (2023)  (cross-doc temporal)
FIXTURE = [
    # --- multi-hop chain ---
    ("Aurelia Systems", "Aurelia Systems to Acquire Nimbus Robotics", "2023-02-14",
     "Aurelia Systems said on Tuesday it had agreed to acquire Nimbus Robotics, a warehouse "
     "automation supplier, in a cash-and-stock transaction. Aurelia Systems will absorb the "
     "Nimbus Robotics engineering organisation and its autonomous navigation portfolio. "
     "Executives at Aurelia Systems described the deal as the company's largest acquisition "
     "to date and said Nimbus Robotics would continue operating under its own brand through "
     "the end of the year."),
    ("Nimbus Robotics", "Nimbus Robotics unveils PathSense navigation stack", "2023-03-02",
     "Nimbus Robotics has developed PathSense, a navigation stack that fuses lidar and wheel "
     "odometry for indoor fleets. Nimbus Robotics said PathSense was built in-house over "
     "eighteen months and is now deployed across pilot warehouses. PathSense is the first "
     "product Nimbus Robotics has shipped since the company entered acquisition talks."),
    ("Veritas Capital Partners", "Veritas Capital Partners leads round in Aurelia Systems", "2023-04-11",
     "Veritas Capital Partners has invested in Aurelia Systems, taking a minority stake in the "
     "industrial software vendor. Veritas Capital Partners said the investment in Aurelia Systems "
     "would fund integration work following recent acquisitions. It is the second time Veritas "
     "Capital Partners has backed a company in the warehouse automation segment this year."),
    # --- multi-hop qua person ---
    ("Halcyon Labs", "Halcyon Labs founded by former Aurelia engineer Mira Sandoval", "2023-01-20",
     "Halcyon Labs was founded by Mira Sandoval, who previously worked at Aurelia Systems as a "
     "principal engineer on control software. Mira Sandoval founded Halcyon Labs after leaving "
     "Aurelia Systems in 2022. Halcyon Labs is building simulation tooling for robot fleets."),
    ("Halcyon Labs", "Veritas Capital Partners backs Halcyon Labs seed round", "2023-05-09",
     "Veritas Capital Partners has invested in Halcyon Labs, the simulation startup founded by "
     "Mira Sandoval. Veritas Capital Partners said Halcyon Labs would use the capital to expand "
     "its engineering team. Halcyon Labs declined to disclose the valuation."),
    # --- cross-doc temporal shift ---
    ("Orion Cloud", "Orion Cloud partners with Aurelia Systems on edge deployment", "2022-11-30",
     "Orion Cloud has partnered with Aurelia Systems to deliver edge deployment tooling for "
     "factory customers. Under the partnership Orion Cloud will host Aurelia Systems workloads "
     "in regional data centres. Orion Cloud said the arrangement was non-exclusive."),
    ("Orion Cloud", "Orion Cloud acquires the Aurelia Systems edge unit", "2023-06-15",
     "Orion Cloud has acquired the edge computing unit of Aurelia Systems, converting a "
     "two-year partnership into an outright purchase. Orion Cloud said the acquired Aurelia "
     "Systems team would join its infrastructure division. The companies had partnered since "
     "late 2022 before Orion Cloud moved to acquire the unit."),
    # --- factoid ---
    ("Delphi AI", "Delphi AI names Priya Raghunathan chief executive", "2023-03-28",
     "Delphi AI has appointed Priya Raghunathan as chief executive officer, effective "
     "immediately. Priya Raghunathan leads Delphi AI after eight years at a rival natural "
     "language processing vendor. Delphi AI said Priya Raghunathan would focus on enterprise "
     "deployments."),
    # --- MERGE_MANUAL: dùng đúng chuỗi có trong MANUAL_ALIASES của notebook ---
    ("Microsoft", "Microsoft Corp expands Azure partnership with Aurelia Systems", "2023-04-02",
     "Microsoft Corp has partnered with Aurelia Systems to bring industrial control workloads to "
     "Azure. Microsoft Corp said the agreement extends an existing relationship. Aurelia Systems "
     "will use Microsoft Corp tooling for model deployment."),
    ("Google", "Google LLC invests in Delphi AI", "2023-05-22",
     "Google LLC has invested in Delphi AI, the natural language platform led by Priya "
     "Raghunathan. Google LLC said Delphi AI would remain independent. The investment gives "
     "Google LLC no board seat, according to Delphi AI."),
    ("Meta", "Meta Platforms partners with Orion Cloud", "2023-02-08",
     "Meta Platforms has partnered with Orion Cloud on regional inference capacity. Meta "
     "Platforms said the deal with Orion Cloud covers three data centre regions."),
    # --- REJECT_GUARD candidates: tên gần giống nhưng là 2 công ty khác nhau ---
    ("Aurelia Therapeutics", "Aurelia Therapeutics reports phase two results", "2023-03-15",
     "Aurelia Therapeutics, a clinical stage biotechnology company unrelated to industrial "
     "software vendors, reported phase two results for its lead candidate. Aurelia "
     "Therapeutics said it would partner with Helix Bio on manufacturing."),
    ("Nimbus Biosciences", "Nimbus Biosciences licenses assay platform", "2023-04-19",
     "Nimbus Biosciences has partnered with Helix Bio to license an assay platform. Nimbus "
     "Biosciences is a diagnostics company and is not affiliated with robotics suppliers."),
    # --- hub: nhiều article cùng trỏ vào Aurelia Systems để đẩy degree lên ---
    ("Aurelia Systems", "Aurelia Systems Inc uses PathSense in retrofit programme", "2023-07-04",
     "Aurelia Systems Inc has begun using PathSense across retrofit deployments. Aurelia "
     "Systems Inc said the navigation stack developed by Nimbus Robotics cut commissioning "
     "time. Aurelia Systems Inc partnered with Orion Cloud on hosting for the programme."),
    ("Aurelia Systems", "Aurelia Systems Corporation leads consortium with Delphi AI", "2023-08-01",
     "Aurelia Systems Corporation has partnered with Delphi AI on a shared research "
     "consortium. Aurelia Systems Corporation said Delphi AI would supply language models "
     "while Aurelia Systems Corporation contributes control software."),
    ("Helix Bio", "Helix Bio partners with Aurelia Therapeutics and Nimbus Biosciences", "2023-06-01",
     "Helix Bio has partnered with Aurelia Therapeutics and with Nimbus Biosciences on "
     "manufacturing capacity. Helix Bio said both agreements are multi-year."),
]

# ---------------------------------------------------------------------------
# 2. GOLDEN — 5 câu, đủ 3 nhóm, reference_answer suy ra trực tiếp từ FIXTURE
# ---------------------------------------------------------------------------
GOLDEN = [
    dict(
        id="P01", group="factoid", difficulty="easy",
        question="Who was named chief executive of Delphi AI in 2023?",
        reference_answer="Priya Raghunathan.",
        reference_evidence="Delphi AI names Priya Raghunathan chief executive (2023-03-28).",
        evidence_titles=["Delphi AI names Priya Raghunathan chief executive"],
        expected_hops=1,
        seed_entities=["Delphi AI", "Priya Raghunathan"],
    ),
    dict(
        id="P02", group="multi-hop", difficulty="hard",
        question=(
            "Which company founded by a former Aurelia Systems employee later received an "
            "investment from Veritas Capital Partners, and who founded it?"
        ),
        reference_answer=(
            "Halcyon Labs, founded by Mira Sandoval, who previously worked at Aurelia Systems. "
            "Veritas Capital Partners invested in Halcyon Labs in May 2023."
        ),
        reference_evidence=(
            "Halcyon Labs founded by former Aurelia engineer Mira Sandoval (2023-01-20) | "
            "Veritas Capital Partners backs Halcyon Labs seed round (2023-05-09)."
        ),
        evidence_titles=[
            "Halcyon Labs founded by former Aurelia engineer Mira Sandoval",
            "Veritas Capital Partners backs Halcyon Labs seed round",
        ],
        expected_hops=3,
        seed_entities=["Aurelia Systems", "Veritas Capital Partners", "Mira Sandoval"],
    ),
    dict(
        id="P03", group="multi-hop", difficulty="hard",
        question=(
            "Trace the chain from Veritas Capital Partners to a named navigation technology: "
            "which company did it invest in, what did that company acquire, and what did the "
            "acquired company develop?"
        ),
        reference_answer=(
            "Veritas Capital Partners invested in Aurelia Systems; Aurelia Systems agreed to "
            "acquire Nimbus Robotics; Nimbus Robotics developed the PathSense navigation stack."
        ),
        reference_evidence=(
            "Veritas Capital Partners leads round in Aurelia Systems (2023-04-11) | Aurelia "
            "Systems to Acquire Nimbus Robotics (2023-02-14) | Nimbus Robotics unveils "
            "PathSense navigation stack (2023-03-02)."
        ),
        evidence_titles=[
            "Veritas Capital Partners leads round in Aurelia Systems",
            "Aurelia Systems to Acquire Nimbus Robotics",
            "Nimbus Robotics unveils PathSense navigation stack",
        ],
        expected_hops=3,
        seed_entities=["Veritas Capital Partners", "Aurelia Systems", "Nimbus Robotics"],
    ),
    dict(
        id="P04", group="cross-doc", difficulty="hard",
        question=(
            "How did the relationship between Orion Cloud and Aurelia Systems change between "
            "2022 and 2023? Cite both dates."
        ),
        reference_answer=(
            "It escalated from partnership to acquisition: on 2022-11-30 Orion Cloud partnered "
            "with Aurelia Systems on edge deployment tooling, and on 2023-06-15 Orion Cloud "
            "acquired the Aurelia Systems edge computing unit, converting the partnership into "
            "an outright purchase."
        ),
        reference_evidence=(
            "Orion Cloud partners with Aurelia Systems on edge deployment (2022-11-30) | "
            "Orion Cloud acquires the Aurelia Systems edge unit (2023-06-15)."
        ),
        evidence_titles=[
            "Orion Cloud partners with Aurelia Systems on edge deployment",
            "Orion Cloud acquires the Aurelia Systems edge unit",
        ],
        expected_hops=2,
        seed_entities=["Orion Cloud", "Aurelia Systems"],
    ),
    dict(
        id="P05", group="cross-doc", difficulty="medium",
        question=(
            "Which organisations partnered with Aurelia Systems across the corpus, and is "
            "Aurelia Therapeutics one of them?"
        ),
        reference_answer=(
            "Aurelia Systems partnered with Orion Cloud, Microsoft and Delphi AI. Aurelia "
            "Therapeutics is a separate clinical-stage biotechnology company and is not one of "
            "Aurelia Systems' partners; it partnered with Helix Bio."
        ),
        reference_evidence=(
            "Orion Cloud partners with Aurelia Systems (2022-11-30) | Microsoft Corp expands "
            "Azure partnership with Aurelia Systems (2023-04-02) | Aurelia Systems Corporation "
            "leads consortium with Delphi AI (2023-08-01) | Aurelia Therapeutics reports phase "
            "two results (2023-03-15)."
        ),
        evidence_titles=[
            "Orion Cloud partners with Aurelia Systems on edge deployment",
            "Microsoft Corp expands Azure partnership with Aurelia Systems",
            "Aurelia Systems Corporation leads consortium with Delphi AI",
            "Aurelia Therapeutics reports phase two results",
        ],
        expected_hops=2,
        seed_entities=["Aurelia Systems", "Aurelia Therapeutics"],
    ),
]

N_REAL = 114          # article văn bản thật lấy từ mirror (làm distractor/nhiễu)
DUP_CLONES = 3        # số bản near-duplicate cố ý chèn để test MinHash/LSH


def build_corpus() -> pd.DataFrame:
    rows = []
    for company, title, date, text in FIXTURE:
        rows.append({
            "companyName": company,
            "articleTitle": title,
            "publishedDate": f"{date} 09:00:00",
            "description": text,
            "sourceUrl": f"https://example.test/{title.lower().replace(' ', '-')[:60]}",
        })

    # Near-duplicate cố ý: cùng nội dung, đổi vài từ + đổi title -> exact dedup KHÔNG
    # bắt được, chỉ MinHash/LSH mới bắt. Dùng để chứng minh Challenge A hoạt động.
    base = rows[0]
    for i in range(DUP_CLONES):
        rows.append({
            "companyName": base["companyName"],
            "articleTitle": base["articleTitle"] + f" (syndicated copy {i + 1})",
            "publishedDate": f"2023-02-1{5 + i} 11:30:00",
            "description": base["description"].replace("on Tuesday", "this week")
                                              .replace("largest", "biggest"),
            "sourceUrl": f"https://syndicate{i}.test/aurelia-nimbus",
        })

    if not MIRROR.exists():
        raise SystemExit(
            f"[FAIL] thiếu {MIRROR}. Tải trước:\n"
            "  https://huggingface.co/datasets/pacozaa/tech-company-news-data-dump-clean"
            "/resolve/main/data/train-00000-of-00001.parquet"
        )
    real = pd.read_parquet(MIRROR)
    real = real.assign(_len=real.description.astype(str).str.len())
    real = real.sort_values("_len", ascending=False).head(N_REAL).reset_index(drop=True)
    for i, r in real.iterrows():
        rows.append({
            "companyName": r.companyName,
            "articleTitle": f"{r.companyName} news roundup",
            # Ngày tăng dần, deterministic -> kiểm được thứ tự "cạnh mới nhất".
            "publishedDate": f"2023-{(i % 12) + 1:02d}-{(i % 27) + 1:02d} 08:00:00",
            "description": str(r.description),
            "sourceUrl": f"https://mirror.test/row{i}",
        })
    return pd.DataFrame(rows)


def build_golden(corpus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    title_to_row = {t: i for i, t in enumerate(corpus["articleTitle"])}

    simple, detailed = [], []
    for g in GOLDEN:
        row_ids = [title_to_row[t] for t in g["evidence_titles"]]
        missing = [t for t in g["evidence_titles"] if t not in title_to_row]
        if missing:
            raise SystemExit(f"[FAIL] golden {g['id']} trỏ tới title không có trong corpus: {missing}")
        simple.append({
            "id": g["id"], "group": g["group"], "difficulty": g["difficulty"],
            "question": g["question"], "reference_answer": g["reference_answer"],
            "reference_evidence": g["reference_evidence"],
        })
        detailed.append({
            **simple[-1],
            "evidence_row_ids_0based": json.dumps(row_ids),
            "expected_hops": g["expected_hops"],
            "seed_entities": json.dumps(g["seed_entities"]),
            "source_scope": "preflight-fixture",
        })
    return pd.DataFrame(simple), pd.DataFrame(detailed)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    corpus = build_corpus()
    golden, golden_detailed = build_golden(corpus)

    corpus.to_csv(OUT / "corpus.csv", index=False)
    golden.to_csv(OUT / "golden.csv", index=False)
    golden_detailed.to_csv(OUT / "golden_detailed.csv", index=False)

    print(f"corpus.csv          : {len(corpus)} article, cột = {list(corpus.columns)}")
    print(f"  fixture           : {len(FIXTURE)} + {DUP_CLONES} near-duplicate")
    print(f"  real text (mirror) : {N_REAL}")
    print(f"golden.csv          : {len(golden)} câu — {golden.group.value_counts().to_dict()}")
    ev = sorted({r for s in golden_detailed.evidence_row_ids_0based for r in json.loads(s)})
    print(f"evidence row ids    : {ev}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
