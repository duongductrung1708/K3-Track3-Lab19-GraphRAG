"""Kiểm tra quyền tải dataset gated trên Hugging Face và chỉ ra CHÍNH XÁC chỗ sai.

Chạy:  .venv/Scripts/python.exe tools/check_hf_access.py
       .venv/Scripts/python.exe tools/check_hf_access.py --wait   # poll tới khi có quyền

Vì sao cần: `load_dataset(...)` chỉ báo "is a gated dataset" chung chung, không phân biệt
được 3 nguyên nhân khác nhau (token sai / account chưa Agree / token thiếu scope gated).
Script này tách rõ từng nguyên nhân bằng cách gọi lần lượt whoami -> metadata -> data file.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
REPO = "HackerNoon/tech-company-news-data-dump"
DATA_FILE = "cleanedCompanyNews.csv"
GATE_URL = f"https://huggingface.co/datasets/{REPO}"


def read_dotenv() -> dict:
    path = ROOT / ".env"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def check(token: str) -> tuple[bool, str]:
    """-> (ok, chẩn đoán). ok=True nghĩa là tải được file data."""
    h = {"Authorization": f"Bearer {token}"}

    # 1. Token còn sống? Có scope gated global không?
    r = requests.get("https://huggingface.co/api/whoami-v2", headers=h, timeout=30)
    if r.status_code != 200:
        return False, f"TOKEN CHẾT (whoami {r.status_code}). Tạo token mới tại https://huggingface.co/settings/tokens"
    me = r.json()
    auth = me.get("auth", {}).get("accessToken", {}) or {}
    fine = auth.get("fineGrained") or {}
    global_scopes = fine.get("global") or []
    has_gated_scope = any("gated" in str(s) for s in global_scopes)
    print(f"  account   : {me.get('name')}")
    print(f"  token role: {auth.get('role')}  global_scopes={global_scopes or '[]'}")

    # 2. Đọc được data file chưa? (metadata public nên không dùng để kết luận)
    r = requests.get(
        f"https://huggingface.co/datasets/{REPO}/resolve/main/{DATA_FILE}",
        headers={**h, "Range": "bytes=0-200"}, timeout=30,
    )
    if r.status_code in (200, 206):
        return True, "OK — tải được data file."

    msg = r.headers.get("x-error-message", "")
    if "not in the authorized list" in msg:
        why = [
            f"CHƯA CÓ QUYỀN ({r.status_code}). Phải làm ĐỦ CẢ 2 việc — thiếu 1 việc là vẫn 403:",
            "",
            "  [1] NHẬN GATE (bắt buộc, không có cách nào bỏ qua)",
            f"      Mở {GATE_URL}",
            f"      Đăng nhập đúng account '{me.get('name')}' -> bấm nút"
            " 'Agree and access repository' ngay đầu trang.",
            "      Dataset này gated=auto nên được duyệt TỨC THÌ, không phải chờ chủ repo.",
            "",
            "  [2] TOKEN ĐỌC ĐƯỢC GATED REPO",
        ]
        if auth.get("role") == "fineGrained" and not has_gated_scope:
            why += [
                f"      Token hiện tại là fineGrained, scope = {global_scopes or '[]'}"
                " -> KHÔNG có quyền đọc gated repo.",
                "      ('discussion.write' / 'post.write' là quyền bình luận, KHÔNG liên quan.)",
                "",
                "      → CÁCH DỄ NHẤT: https://huggingface.co/settings/tokens -> 'Create new token'",
                "        -> chọn Token type = **Read** (không phải Fine-grained) -> Create.",
                "        Token 'Read' tự động đọc được mọi gated repo mà bạn đã Agree.",
                "",
                "      → Nếu vẫn muốn dùng Fine-grained: sửa token, kéo xuống mục 'Repos'",
                "        và tick 'Read access to contents of all public gated repos you can access'.",
                "",
                "      Xong thì dán token mới vào .env:  HF_TOKEN=hf_xxx",
            ]
        else:
            why.append("      Token đã đủ scope đọc gated repo -> chỉ còn thiếu bước [1] ở trên.")
        return False, "\n".join(why)
    return False, f"Lỗi khác ({r.status_code}): {msg or r.text[:200]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", action="store_true", help="poll 15s/lần tới khi có quyền")
    ap.add_argument("--timeout", type=int, default=1800, help="giây tối đa khi --wait")
    args = ap.parse_args()

    token = read_dotenv().get("HF_TOKEN", "")
    if not token:
        print("✗ Không thấy HF_TOKEN trong .env")
        return 2

    deadline = time.time() + args.timeout
    while True:
        # .env có thể được sửa giữa lúc poll -> đọc lại mỗi vòng.
        token = read_dotenv().get("HF_TOKEN", token)
        ok, diag = check(token)
        if ok:
            print(f"✅ {diag}")
            return 0
        print(f"✗ {diag}")
        if not args.wait or time.time() > deadline:
            return 1
        print("  ...chờ 15s rồi thử lại (Ctrl+C để dừng)\n")
        time.sleep(15)


if __name__ == "__main__":
    sys.exit(main())
