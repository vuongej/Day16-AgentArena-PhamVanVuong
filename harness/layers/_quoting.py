"""Kiểm tra trích dẫn theo ĐÚNG luật của scorer — dùng chung cho
`citation_checker` (§11) và `critic` (§2).

Hai lớp đó cùng phải trả lời một câu hỏi: "câu này có phải trích nguyên
văn MỘT DÒNG của tài liệu kia không?". `arena.scorer._supports` trả lời
bằng cách chuẩn hoá NFC + casefold + gộp khoảng trắng rồi so khớp
substring TRONG PHẠM VI MỘT DÒNG. Viết lại đúng phép chuẩn hoá đó ở đây
thay vì gọi hàm private của scorer: `arena/` là scaffold đông lạnh, chỉ
đọc, và một helper 5 dòng thì rẻ hơn một phụ thuộc vào API riêng tư.

`in doc.body` (cả khối) là SAI — nó nhận cả câu vắt qua hai dòng, thứ mà
scorer chấm `HALLUCINATED`.
"""

from __future__ import annotations

import re
import unicodedata

_WS_RE = re.compile(r"\s+")


def norm(text: str) -> str:
    """Dạng chuẩn hoá mà mọi phép so khớp của scorer chạy trên đó."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text).casefold()).strip()


def quotes_a_line(body: str, text: str) -> bool:
    """`text` có nằm gọn trong MỘT dòng của `body` không?"""
    needle = norm(text)
    if not needle:
        return False
    return any(needle in norm(line) for line in body.splitlines())


def source_doc(corpus, observed_text: str, text: str):
    """Tài liệu ĐÃ QUAN SÁT đầu tiên thực sự chứa câu này, hoặc None.

    `doc.body in observed_text` nghĩa là "tài liệu này đã về NGUYÊN VẸN từ
    một lần fetch sạch": một snippet của search hay một bản bị cắt không
    tính, và trích vào tài liệu chưa từng đọc bị chấm `UNRETRIEVED`.
    """
    if corpus is None:
        return None
    for doc in corpus.docs:
        if doc.body in observed_text and quotes_a_line(doc.body, text):
            return doc
    return None
