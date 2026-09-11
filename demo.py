"""
Kịch bản demo toàn bộ project — chạy lần lượt cả 11 hàm của template.py.

Chạy:
    python demo.py            # demo đầy đủ, có gọi API thật (cần .env)
    python demo.py --offline  # chỉ các phần không cần mạng

Demo này KHÔNG nằm trong thang điểm. Nó chỉ để bạn nhìn thấy từng hàm
hoạt động ra sao trên dữ liệu thật.
"""

import builtins
import sys
import time

from template import (
    OPENAI_MINI_MODEL,
    OPENAI_MODEL,
    batch_compare,
    call_openai,
    call_openai_mini,
    chat_with_system_prompt,
    compare_models,
    count_tokens,
    estimate_cost,
    format_comparison_table,
    retry_with_backoff,
    run_assistant,
    streaming_chatbot,
)

OFFLINE = "--offline" in sys.argv
PROMPT = "Việt Nam có bao nhiêu tỉnh thành? Trả lời ngắn gọn."


def buoc(so: str, ten: str, ham: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"BƯỚC {so} — {ten}")
    print(f"{'-' * 72}")
    print(f"hàm: {ham}")
    print()


def scripted_input(cac_luot):
    """Thay thế input() bằng một kịch bản gõ phím định sẵn."""
    it = iter(cac_luot)

    def _fake(prompt=""):
        try:
            msg = next(it)
        except StopIteration:
            msg = "quit"
        print(f"Bạn: {msg}")
        return msg

    return _fake


# ---------------------------------------------------------------------------
print(f"\n{'#' * 72}")
print("# DEMO TOÀN BỘ PROJECT — K4 Ngày 1: Khám Phá LLM API")
print(f"# model lớn: {OPENAI_MODEL}")
print(f"# model nhỏ: {OPENAI_MINI_MODEL}")
print(f"# chế độ   : {'OFFLINE (bỏ qua phần gọi API)' if OFFLINE else 'ĐẦY ĐỦ (gọi API thật)'}")
print(f"{'#' * 72}")

bat_dau = time.perf_counter()

# ===========================================================================
# PART 1
# ===========================================================================
if not OFFLINE:
    buoc("1.1", "Gọi model lớn, đo độ trễ", "call_openai()")
    text_lon, lat_lon = call_openai(PROMPT, max_tokens=1500)
    print(f"[{lat_lon:.2f}s] {text_lon.strip()[:300]}")

    buoc("1.2", "Gọi model nhỏ với cùng prompt", "call_openai_mini()")
    text_nho, lat_nho = call_openai_mini(PROMPT, max_tokens=1500)
    print(f"[{lat_nho:.2f}s] {text_nho.strip()[:300]}")
    print(f"\n→ Model nhỏ nhanh hơn {lat_lon / lat_nho:.1f} lần.")

    buoc("1.3", "So sánh hai model trên cùng một prompt", "compare_models()")
    kq = compare_models(PROMPT)
    for khoa, gia_tri in kq.items():
        if isinstance(gia_tri, float):
            print(f"  {khoa:22s} {gia_tri:.6f}")
        else:
            print(f"  {khoa:22s} {str(gia_tri).strip()[:120]}")
    print("\n→ Lưu ý: compare_models dùng max_tokens mặc định 256, nên model lớn")
    print("  có thể bị cắt vì nó tiêu ngân sách token vào suy luận nội bộ.")

# ===========================================================================
# PART 2
# ===========================================================================
if not OFFLINE:
    buoc("2.1", "Cùng câu hỏi, hai persona khác nhau", "chat_with_system_prompt()")
    cau_hoi = "Giải thích blockchain là gì?"
    personas = {
        "Giáo viên tiểu học": "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi.",
        "Chuyên gia tài chính": "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật.",
    }
    luu = {}
    for ten, persona in personas.items():
        tra_loi, do_tre = chat_with_system_prompt(persona, cau_hoi, max_tokens=1500)
        tra_loi = tra_loi.strip()
        luu[ten] = tra_loi
        print(f"--- {ten} ({do_tre:.2f}s, {len(tra_loi.split())} từ) ---")
        print(" ".join(tra_loi.split())[:400], "...\n")
    print("→ Độ dài gần bằng nhau, nhưng từ vựng và ví dụ khác hẳn.")

buoc("2.2", "Đếm token thật so với ước lượng theo số từ", "count_tokens()")
doan_vi = ("Trí tuệ nhân tạo đang thay đổi cách con người làm việc mỗi ngày. "
           "Những mô hình ngôn ngữ lớn có thể đọc hiểu văn bản, tóm tắt tài liệu dài "
           "và viết mã nguồn thay cho lập trình viên.")
doan_en = ("Artificial intelligence is changing the way people work every day. Large "
           "language models can read text, summarise long documents and write source "
           "code for programmers.")
print(f"  {'ngôn ngữ':12s} {'số từ':>7s} {'tiktoken':>9s} {'từ/0.75':>9s} {'token/từ':>9s}")
for nhan, doan in (("Tiếng Việt", doan_vi), ("Tiếng Anh", doan_en)):
    so_tu = len(doan.split())
    tok = count_tokens(doan, model="gpt-4o")
    print(f"  {nhan:12s} {so_tu:7d} {tok:9d} {so_tu / 0.75:9.1f} {tok / so_tu:9.2f}")
print("\n  Model lạ (không có bảng mã) → rơi về fallback len(text)//4:")
print(f"    count_tokens(doan_vi, model='khong-ton-tai') = {count_tokens(doan_vi, model='khong-ton-tai')}")

buoc("2.3", "Tách bạch chi phí input và output", "estimate_cost()")
cp = estimate_cost("Giải thích blockchain là gì?", doan_vi, model="gpt-4o")
for khoa, gia_tri in cp.items():
    print(f"  {khoa:16s} {gia_tri:.8f}" if isinstance(gia_tri, float) else f"  {khoa:16s} {gia_tri}")
print(f"\n→ Output đắt hơn input {cp['output_cost'] / cp['input_cost']:.1f} lần "
      f"dù chỉ nhiều hơn {cp['output_tokens'] - cp['input_tokens']} token.")

# ===========================================================================
# PART 3
# ===========================================================================
buoc("3.2", "Thử lại có backoff khi gặp lỗi tạm thời", "retry_with_backoff()")
dem = {"n": 0}


def ham_chap_chon():
    dem["n"] += 1
    print(f"    lần gọi {dem['n']} lúc {time.perf_counter() - moc:.3f}s", end="")
    if dem["n"] < 3:
        print("  → lỗi tạm thời")
        raise ConnectionError("server đang quá tải")
    print("  → thành công")
    return "dữ liệu trả về"


moc = time.perf_counter()
ket_qua = retry_with_backoff(ham_chap_chon, max_retries=3, base_delay=0.2)
print(f"  kết quả: {ket_qua!r}")
print("\n→ Khoảng chờ tăng gấp đôi: 0.2s rồi 0.4s. Đó là exponential backoff.")

print("\n  Trường hợp hỏng vĩnh viễn — lỗi gốc phải được ném ra nguyên vẹn:")
try:
    retry_with_backoff(lambda: (_ for _ in ()).throw(ValueError("hỏng hẳn")),
                       max_retries=2, base_delay=0.05)
except ValueError as e:
    print(f"    bắt được đúng {type(e).__name__}: {e}")

if not OFFLINE:
    buoc("3.1", "Chatbot streaming, kịch bản 2 lượt rồi thoát", "streaming_chatbot()")
    that_input = builtins.input
    builtins.input = scripted_input([
        "Chào bạn, bạn là ai?",
        "Vừa rồi mình hỏi gì?",
        "quit",
    ])
    try:
        streaming_chatbot()
    finally:
        builtins.input = that_input
    print("→ Lượt 2 trả lời được là nhờ history giữ lại ngữ cảnh lượt 1.")

# ===========================================================================
# PART 4
# ===========================================================================
if not OFFLINE:
    buoc("4", "Trợ lý CLI hoàn chỉnh — ghép tất cả lại", "run_assistant()")
    persona = ("Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt. "
               "Nếu không chắc chắn, hãy nói rõ là chưa chắc thay vì đoán.")
    thong_ke = run_assistant(
        persona=persona,
        get_input=scripted_input([
            "Token trong LLM là gì? Trả lời 2 câu.",
            "Vậy nó khác gì với từ?",
            "quit",
        ]),
    )
    print("\n--- Thống kê phiên chat ---")
    for khoa, gia_tri in thong_ke.items():
        if khoa == "history":
            print(f"  {khoa:14s} {len(gia_tri)} message (tối đa 6)")
        else:
            print(f"  {khoa:14s} {gia_tri}")
    print("\n  History còn lại:")
    for msg in thong_ke["history"]:
        print(f"    [{msg['role']:9s}] {' '.join(msg['content'].split())[:90]}")

# ===========================================================================
# BONUS
# ===========================================================================
if not OFFLINE:
    buoc("B", "So sánh nhiều prompt và in thành bảng",
         "batch_compare() + format_comparison_table()")
    ket = batch_compare([
        "Thủ đô của Việt Nam là gì?",
        "Kể tên một món ăn Việt Nam nổi tiếng.",
    ])
    print(format_comparison_table(ket))

print(f"\n{'#' * 72}")
print(f"# DEMO HOÀN TẤT trong {time.perf_counter() - bat_dau:.1f}s")
print(f"{'#' * 72}\n")
