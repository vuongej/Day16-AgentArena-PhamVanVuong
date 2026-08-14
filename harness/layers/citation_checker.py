"""LỚP `citation_checker` — bài giảng Day 16, §11 (Grounding & Citations).

NHIỆM VỤ: chỉ cần MỘT tài liệu gắn nhãn `lookalike` hoặc `outdated` lọt
vào bằng chứng là mô hình neo TOÀN BỘ claim vào đúng tài liệu trông có vẻ
"chính thống" đó — dù mỗi câu được lấy nguyên văn từ một tài liệu khác.
Câu thì thật, trích dẫn thì sai. Đây là kiểu sai nguy hiểm nhất trong RAG
vì báo cáo đọc vào vẫn rất thuyết phục.

TÍN HIỆU (chính xác, không cần đoán):

    claim["text"] KHÔNG khớp NGUYÊN VĂN một DÒNG nào trong
    corpus.get(claim["doc_id"]).body
    nhưng CHÍNH câu đó CÓ trong bằng chứng agent đã quan sát

Chú ý chữ DÒNG: kiểm tra `claim["text"] in doc.body` (cả khối, không
tách dòng) là SAI — scorer chỉ nhận trích dẫn khớp nguyên văn MỘT DÒNG
(xem "ĐƯỢC PHÉP VÀ KHÔNG ĐƯỢC PHÉP" ngay dưới đây). `in doc.body` coi
một câu vắt qua hai dòng là hợp lệ, trong khi scorer thì không — tín
hiệu kiểu đó khiến bạn giữ nguyên một trích dẫn mà scorer vẫn chấm
`HALLUCINATED`.

Vế thứ hai mới là phần quan trọng: nó tách việc của bạn khỏi việc của
`critic` (§2). Câu có trong bằng chứng nhưng gắn sai tài liệu -> GẮN LẠI
(việc của bạn). Câu không có trong bằng chứng nào -> BỊA, để `critic` xoá.
Hai điều kiện loại trừ nhau nên hai lớp không giành điểm của nhau.

ĐƯỢC PHÉP VÀ KHÔNG ĐƯỢC PHÉP:
  * ĐƯỢC: đổi `claim["doc_id"]`, cập nhật `report["citations"]`.
  * KHÔNG: sửa `claim["text"]`. Scorer chỉ cho điểm khi câu là trích dẫn
    nguyên văn của MỘT DÒNG trong tài liệu được trích VÀ đúng là chữ mô
    hình đã viết. Thêm dấu chấm, đổi dấu nháy, "chuẩn hoá" khoảng trắng,
    hay vá lại câu bị cắt bằng nội dung lấy từ corpus đều làm mất cả hai
    điều kiện cùng lúc (đo được: -40 điểm).

CHỈ ĐƯỢC GẮN VÀO TÀI LIỆU ĐÃ QUAN SÁT. Trích một tài liệu mà lượt chạy
chưa từng đọc bị chấm `UNRETRIEVED`. Vì vậy hãy tìm nguồn trong
`ctx.observed_text`, đừng quét cả corpus rồi gắn bừa: điều kiện
`doc.body in ctx.observed_text` nghĩa là "tài liệu này đã về nguyên vẹn
từ một lần fetch sạch" — một đoạn snippet hay một bản bị cắt không tính.

CÔNG CỤ CÓ SẴN:
    ctx.observed_text  -> toàn bộ quan sát agent đã thấy, nối lại
    ctx.corpus.get(doc_id) -> Doc | None
    ctx.corpus.docs    -> danh sách Doc (doc_id, title, body); qua
                          `ctx.corpus`, `Doc.tags` LUÔN RỖNG — CẢ Ở VÒNG
                          LUYỆN TẬP LẪN VÒNG CHẤM ĐIỂM, vì corpus mà code
                          của bạn cầm bị gỡ nhãn bẫy ('outdated',
                          'contradiction', 'injection'…) ngay khi runner
                          dựng lên nó, không phải chỉ lúc chấm điểm. Đọc
                          nhãn là tra bảng chứ không phải kỹ năng lab này
                          chấm. Ở vòng LUYỆN TẬP seed 42 thì file TRÊN ĐĨA
                          `data/corpus/*.json` (khác với `ctx.corpus`)
                          vẫn có nhãn: hard-code được từ đó, và điều đó
                          được nói thẳng ra ở đây thay vì giấu đi.

Cài đặt:  ReActAgent(..., middleware=[..., CitationChecker(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

from harness.layers._quoting import quotes_a_line, source_doc
from harness.middleware import Middleware

#: Siết lại quy tắc 1 của `ARENA_SYSTEM_PROMPT` cho model THẬT.
#:
#: Prompt đóng băng chỉ nói "một câu TRÍCH NGUYÊN VĂN". Đo trên
#: gpt-5.6-luna: nó trích ĐÚNG nhưng CẮT CỤT — claim vẫn được chấm
#: SUPPORTED, doc_id vẫn đúng, precision vẫn 1.0, nhưng đoạn nó chép
#: không phủ hết key_terms của required_fact nên recall = 0. Đó là 5/9
#: brief có G 0.00 ở đường thật.
#:
#: Chữa sau khi model đã viết là KHÔNG THỂ: kéo dài `claim["text"]` là
#: sửa chữ, mất provenance ngay. Nên chỗ duy nhất sửa được là TRƯỚC khi
#: model viết.
#:
#: CHỈ nối vào message `system`. `arena.model._first_user_content` lấy
#: message USER cuối cùng trước lượt assistant đầu tiên làm câu hỏi của
#: brief, nên chèn vào user là biến chính lời nhắc thành truy vấn tìm
#: kiếm cho cả lượt chạy — không một dòng lỗi nào báo.
QUOTE_RULE = """

QUY TẮC TRÍCH DẪN (bổ sung, bắt buộc):
1a. Mỗi "text" trong "claims" phải sao chép TRỌN VẸN MỘT DÒNG của tài liệu — từ đầu
    dòng đến hết dòng đó. KHÔNG cắt ngắn, KHÔNG dừng giữa câu, KHÔNG thay phần còn
    lại bằng "...".
1b. Chép đúng từng ký tự: giữ nguyên dấu câu, chữ hoa, dấu nháy và khoảng trắng.
    Chỉ cần thêm một dấu chấm là claim đó mất giá trị hoàn toàn.
1c. KHÔNG ghép nội dung của hai dòng khác nhau, hoặc của hai tài liệu khác nhau,
    vào cùng một "text".
1d. Ưu tiên chọn dòng nào trả lời TRỰC TIẾP câu hỏi của brief, và chép trọn dòng ấy.

ĐÃ THỬ VÀ ĐÃ BỎ — đừng thêm lại: một khối "quy tắc truy xuất" (tìm lại bằng từ khoá
khác, bỏ qua chi tiết bối cảnh trong câu hỏi) đo được 63.20 trên 3 lượt real so với
63.93 khi không có nó, và kéo mock từ 81.71 xuống 81.44. Nút thắt của pub-08/09 là
tài liệu bắt buộc không nằm trong top-10 BM25 của chính câu hỏi — prompt không chữa
được, và bảng đồng nghĩa chữa được thì chính là hard-code bộ công khai."""


class CitationChecker(Middleware):
    """Trỏ mỗi claim về đúng tài liệu thật sự chứa câu đó."""

    name = "citation_checker"

    def before_model(self, ctx, messages):
        out, patched = [], False
        for message in messages:
            if not patched and message.get("role") == "system":
                out.append({**message, "content": message.get("content", "") + QUOTE_RULE})
                patched = True
            else:
                out.append(message)
        # Không có message `system` (endpoint bỏ system): để nguyên. Nối
        # vào user ở đây đắt hơn nhiều so với việc bỏ qua lời nhắc.
        return out if patched else messages

    def after_agent(self, ctx, report):
        claims = report.get("claims")
        if not isinstance(claims, list) or not claims or ctx.corpus is None:
            return report
        observed = ctx.observed_text
        fixed = 0
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            text = claim.get("text")
            if not isinstance(text, str) or not text:
                continue
            cited = ctx.corpus.get(claim.get("doc_id"))
            if cited is not None and quotes_a_line(cited.body, text):
                continue  # trích dẫn đã đúng
            truth = source_doc(ctx.corpus, observed, text)
            if truth is not None:
                # CHỈ đổi doc_id. Sửa text là mất cả provenance lẫn hỗ trợ.
                claim["doc_id"] = truth.doc_id
                fixed += 1
            # Không tìm được nguồn nào -> câu bịa, để `critic` xoá.
        ctx.state["citations_reattributed"] = fixed
        report["citations"] = _cited_ids(claims)
        return report


def _cited_ids(claims) -> list:
    ids = {
        c["doc_id"]
        for c in claims
        if isinstance(c, dict) and isinstance(c.get("doc_id"), str) and c["doc_id"]
    }
    return sorted(ids)
