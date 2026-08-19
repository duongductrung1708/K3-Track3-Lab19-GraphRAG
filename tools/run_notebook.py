"""Chạy notebook headless đúng nghĩa "Restart & Run All" và LƯU output vào file.

Chạy:  .venv/Scripts/python.exe tools/run_notebook.py
Tuỳ chọn:
    --timeout 3600      giới hạn giây cho MỖI cell (LLM call nên để cao)
    --start-from 20      chỉ chạy từ cell 20 trở đi (giữ output cũ của cell trước)
    --stop-after 8       chạy tới hết cell 8 rồi dừng (chạy pipeline theo chặng)
    --dry-run            chỉ kiểm tra kernel + liệt kê cell, không chạy
    --allow-errors       chạy tiếp khi cell lỗi (để thu hết lỗi trong 1 lượt)

Vì sao cần script này thay vì bấm Run All trên Jupyter UI:
  - RUBRIC đòi notebook nộp kèm output; chạy headless ghi output trực tiếp vào .ipynb.
  - Pipeline dài (~200 lệnh gọi LLM). Script SAVE SAU MỖI CELL nên mất mạng /
    rate limit giữa đường vẫn giữ được toàn bộ output đã có, không phải chạy lại.
  - Chạy được trong terminal/agent, không cần mở browser.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb"
LOG = ROOT / "outputs" / "run_notebook.log"


def code_cells(nb) -> list[int]:
    return [i for i, c in enumerate(nb.cells) if c.cell_type == "code"]


def preview(cell) -> str:
    for line in cell.source.splitlines():
        line = line.strip()
        if line and not line.startswith("#@title"):
            return line[:70]
    return (cell.source.splitlines() or [""])[0][:70]


def title(cell) -> str:
    first = (cell.source.splitlines() or [""])[0].strip()
    return first[len("#@title"):].strip() if first.startswith("#@title") else preview(cell)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=3600, help="giây cho mỗi cell")
    ap.add_argument("--start-from", type=int, default=0, help="index cell bắt đầu")
    ap.add_argument("--stop-after", type=int, default=None, help="index cell cuối (bao gồm)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-errors", action="store_true")
    args = ap.parse_args()

    nb = nbformat.read(NB, as_version=4)
    codes = code_cells(nb)
    print(f"Notebook: {NB.name}  ({len(nb.cells)} cell, {len(codes)} code cell)")

    if args.start_from:
        # Cell trước start_from bị đánh dấu skip: nbclient không có API "chạy 1 phần",
        # nên ta tạm biến chúng thành raw cell rồi phục hồi sau khi chạy.
        print(f"Chỉ chạy từ cell {args.start_from} (giữ nguyên output cell trước đó).")
    if args.stop_after is not None:
        print(f"Dừng sau cell {args.stop_after}.")

    if args.dry_run:
        for i in codes:
            n_out = len(nb.cells[i].get("outputs") or [])
            print(f"  {i:03d} out={n_out:<3d} {title(nb.cells[i])}")
        return 0

    skipped = []
    for i in codes:
        if i < args.start_from or (args.stop_after is not None and i > args.stop_after):
            nb.cells[i].cell_type = "raw"          # nbclient bỏ qua raw cell
            skipped.append(i)

    client = NotebookClient(
        nb,
        timeout=args.timeout,
        kernel_name="python3",
        allow_errors=args.allow_errors,
        resources={"metadata": {"path": str(ROOT)}},   # cwd = project root
    )

    started = time.time()
    lines: list[str] = []

    def save() -> None:
        for i in skipped:                            # phục hồi trước khi ghi ra file
            nb.cells[i].cell_type = "code"
        nbformat.write(nb, NB)
        for i in skipped:
            nb.cells[i].cell_type = "raw"

    def on_cell_start(cell, cell_index, **kw):
        if cell.cell_type != "code":
            return
        msg = f"[{time.time()-started:7.1f}s] ▶ cell {cell_index:03d}  {title(cell)}"
        print(msg, flush=True)
        lines.append(msg)
        cell.metadata["_t0"] = time.time()

    def on_cell_end(cell, cell_index, **kw):
        if cell.cell_type != "code":
            return
        dur = time.time() - cell.metadata.pop("_t0", time.time())
        err = [o for o in (cell.get("outputs") or []) if o.get("output_type") == "error"]
        flag = f"✗ {err[0]['ename']}" if err else "✓"
        msg = f"[{time.time()-started:7.1f}s] {flag} cell {cell_index:03d} ({dur:.1f}s)"
        print(msg, flush=True)
        lines.append(msg)
        save()                                        # checkpoint sau mỗi cell

    client.on_cell_start = on_cell_start
    client.on_cell_executed = on_cell_end

    rc = 0
    try:
        client.execute()
    except CellExecutionError as e:
        print(f"\n✗ DỪNG vì cell lỗi:\n{str(e)[-2500:]}", flush=True)
        rc = 1
    except KeyboardInterrupt:
        print("\n⚠️  Người dùng ngắt — output tới cell hiện tại đã được lưu.", flush=True)
        rc = 130
    finally:
        save()
        LOG.parent.mkdir(parents=True, exist_ok=True)
        LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")

    done = sum(1 for i in code_cells(nb) if (nb.cells[i].get("outputs") or []))
    print(f"\nTổng {time.time()-started:.1f}s | cell có output: {done}/{len(codes)}")
    print(f"Notebook đã lưu: {NB}")
    print(f"Log: {LOG}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
