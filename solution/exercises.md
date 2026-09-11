# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Mình chạy trên `gemini-3.5-flash` qua endpoint tương thích OpenAI, mỗi mức
> nhiệt độ gọi hai lần để xem có tái lập được không.
>
> Quy luật rõ nhất là **tính tái lập giảm dần khi nhiệt độ tăng**. Ở 0.0, hai
> lần gọi trả về văn bản giống nhau từng chữ, kể cả cách xuống dòng và chỗ in
> đậm. Từ 0.5 trở lên, hai lần gọi bắt đầu khác nhau ở câu mở đầu, ở ký hiệu
> danh sách (lúc đánh số, lúc gạch đầu dòng) và ở việc chọn giữ hay bỏ chi
> tiết nào. Lên 1.0 và 1.5 thì model còn tự thêm phần phụ ngoài yêu cầu: một
> lần chèn thêm sự thật về họ Nguyễn chiếm gần 40% dân số, một lần mở hẳn mục
> "Tặng bạn thêm" rồi bị cắt vì chạm trần token.
>
> Điều bất ngờ là **nhiệt độ không đổi được nội dung cốt lõi**. Cả tám lần gọi,
> từ 0.0 tới 1.5, đều chọn đúng một chủ đề là hang Sơn Đoòng, và đều nhắc lại
> cùng những chi tiết trụ cột: chứa lọt tòa nhà 40 tầng, máy bay Boeing 747
> bay qua được, có hệ thời tiết riêng, ông Hồ Khanh phát hiện năm 1991. Nói
> cách khác, nhiệt độ điều chỉnh cách diễn đạt trên bề mặt chứ không lật được
> lựa chọn bên dưới, khi mà xác suất của một đáp án đã áp đảo phần còn lại.
> Muốn đổi hẳn nội dung thì phải đổi prompt, chẳng hạn yêu cầu tránh chủ đề
> hang động, chứ vặn nhiệt độ là không đủ.
>
> Một quan sát phụ: độ trễ dao động từ 9,0 tới 21,9 giây nhưng **không liên
> quan gì tới nhiệt độ**. Nó phụ thuộc độ dài đầu ra và tải của máy chủ.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Mình chọn **0.2**. Chatbot hỗ trợ khách hàng phải trả lời nhất quán: cùng một
> câu hỏi về chính sách đổi trả thì mọi khách phải nhận cùng một đáp án, nếu
> không thì bộ phận CSKH không bảo vệ được câu trả lời của chính mình. Nhiệt độ
> thấp cũng giảm nguy cơ model bịa ra điều khoản không tồn tại, và làm cho việc
> kiểm thử tự động trở nên khả thi vì đầu ra gần như tái lập được. Mình không
> đặt hẳn 0.0 vì để lại một chút ngẫu nhiên giúp câu chữ bớt máy móc khi model
> diễn đạt lại cùng một nội dung, nhưng nếu là chatbot tra cứu điều khoản thuần
> túy thì 0.0 là lựa chọn đúng hơn.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Khối lượng là 10.000 x 3 = 30.000 lượt gọi mỗi ngày, tương đương 10,5 triệu
> token đầu ra. Với đơn giá output trong `PRICING_PER_1K_TOKENS`, GPT-4o tốn
> **105 USD/ngày (~3.150 USD/tháng)** còn GPT-4o-mini chỉ **6,30 USD/ngày
> (~189 USD/tháng)**, tức GPT-4o **đắt hơn khoảng 16,7 lần** (0,010 so với
> 0,0006 USD cho mỗi 1K token output). Chênh lệch gần 3.000 USD mỗi tháng là
> ngân sách thật, đủ để thuê thêm người.
>
> GPT-4o xứng đáng khi sai một lần là tốn kém hơn nhiều lần số tiền tiết kiệm
> được: sinh mã nguồn đưa thẳng vào pull request, tóm tắt hồ sơ y tế hay hợp
> đồng pháp lý, hoặc suy luận nhiều bước trên dữ liệu tài chính. Ở đó chi phí
> của một câu trả lời sai là bug production hoặc rủi ro pháp lý.
>
> Nên dùng mini cho các tác vụ khối lượng lớn nhưng đơn giản và có thể kiểm
> chứng ngay: phân loại ý định tin nhắn, định tuyến ticket vào đúng hàng đợi,
> trích xuất trường dữ liệu theo khuôn mẫu cố định, hay viết lại câu cho gọn.
> Chiến lược thực tế là phân tầng: mini xử lý mặc định, chỉ leo thang lên
> GPT-4o khi mini trả về độ tin cậy thấp hoặc khi người dùng yêu cầu.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Mình gọi `chat_with_system_prompt` hai lần với cùng câu hỏi "Giải thích
> blockchain là gì?" và chỉ đổi system prompt. Kết quả đo được:
>
> | Persona | Số từ | Token | Độ trễ |
> |---|---:|---:|---:|
> | Giáo viên tiểu học | 354 | 514 | 18,70s |
> | Chuyên gia tài chính | 362 | 538 | 16,71s |
>
> Điều đáng chú ý đầu tiên là **độ dài gần như y hệt nhau**, chênh chưa tới 3%.
> System prompt không đổi lượng chữ mà đổi hoàn toàn chất của chữ.
>
> Từ vựng tách bạch như hai văn bản của hai ngành khác nhau. Bản giáo viên xưng
> hô "cô" và "con yêu", và không dùng một thuật ngữ tiếng Anh nào ngoài hai từ
> Block và Chain, mà cũng dịch ngay ra "khối" và "chuỗi". Bản chuyên gia mở đầu
> bằng "Công nghệ Sổ cái Phân tán (Distributed Ledger Technology)" rồi rải dày
> đặc thuật ngữ nguyên bản: Merkle Root, Nonce, Proof of Work, mã hóa bất đối
> xứng, Trusted Third Party.
>
> Ví dụ minh họa cũng dịch chuyển theo. Giáo viên dựng nguyên một câu chuyện
> lớp học đổi sticker, cả lớp cùng chép vào sổ tay, rồi lấy "sợi dây xích vô
> hình" nối các trang để không ai xé trộm được. Chuyên gia thì mổ xẻ cấu trúc
> một khối thành Block Header và Block Body, và giải thích tính bất biến bằng
> cơ chế băm: đổi một byte ở khối trước là mã băm đổi theo, làm gãy toàn bộ
> chuỗi phía sau.
>
> Rút ra: system prompt không thêm hay bớt kiến thức của model, mà chọn **tầng
> trình bày** cho khối kiến thức đó. Cùng một khái niệm bất biến, một bên diễn
> đạt bằng sợi dây xích, một bên bằng liên kết mã băm. Đây cũng là lý do
> persona phải được gửi lại ở đầu mỗi lượt gọi thay vì nằm trong `history`,
> vì nếu nó trôi mất thì model rơi về giọng mặc định ngay ở lượt kế tiếp.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Mình đo trên một đoạn tiếng Việt 132 từ. `count_tokens` với bộ mã hóa của
> GPT-4o cho **156 token**, trong khi ước lượng `số từ / 0.75` cho **176
> token**, tức ước lượng **cao hơn khoảng 11%**. Điều bất ngờ là ước lượng thô
> lại *thổi phồng* chứ không hụt, vì quy ước 0.75 từ mỗi token vốn được rút ra
> từ tiếng Anh có nhiều từ dài bị tách nhỏ, còn tiếng Việt lại gồm phần lớn là
> âm tiết ngắn nên mỗi từ thường vừa khít một token.
>
> Chuyện tiếng Việt tốn nhiều token hơn vẫn đúng, nhưng vì một lý do khác và
> mức độ phụ thuộc vào bộ mã hóa. Mình so cùng một nội dung ở hai ngôn ngữ:
>
> | | Số từ | Token | Token mỗi từ |
> |---|---:|---:|---:|
> | Tiếng Việt, bộ mã o200k của GPT-4o | 40 | 49 | 1,23 |
> | Tiếng Anh, bộ mã o200k của GPT-4o | 25 | 29 | 1,16 |
> | Tiếng Việt, bộ mã cl100k của GPT-4 cũ | 40 | 84 | 2,10 |
> | Tiếng Anh, bộ mã cl100k của GPT-4 cũ | 25 | 30 | 1,20 |
>
> Đọc bảng này thấy hai nguyên nhân tách bạch. Thứ nhất là cấu trúc ngôn ngữ:
> để diễn đạt cùng một ý, tiếng Việt cần 40 từ còn tiếng Anh chỉ cần 25, nên
> tổng token là 49 so với 29, **nhiều hơn khoảng 69%** dù tỷ lệ token trên mỗi
> từ gần như nhau. Tiếng Việt viết rời từng âm tiết nên đếm theo từ bị đội lên.
>
> Thứ hai là chất lượng bộ mã hóa, và đây mới là phần thay đổi mạnh. Bộ mã
> cl100k của thế hệ GPT-4 cũ phải cắt dấu thanh tiếng Việt thành nhiều mảnh,
> đẩy lên 2,10 token mỗi từ, gần gấp đôi tiếng Anh. Bộ mã o200k của GPT-4o mở
> rộng từ vựng nên nuốt trọn phần lớn âm tiết có dấu, kéo tỷ lệ xuống còn 1,23,
> gần như ngang tiếng Anh. Nói cách khác, thiệt thòi về token của tiếng Việt
> đã giảm rất nhiều ở thế hệ model mới, và bài học rút ra là phải đếm bằng
> tiktoken đúng model đang dùng chứ đừng chép lại hệ số ước lượng cũ.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi con người đang ngồi chờ trước màn hình và câu
> trả lời dài, tức là chatbot hội thoại, trợ lý viết lách hay công cụ sinh mã.
> Ở đó thứ quyết định cảm nhận không phải tổng thời gian mà là thời gian tới
> ký tự đầu tiên: người dùng thấy chữ chạy sau vài trăm mili giây thì chấp nhận
> chờ thêm mười giây, còn nhìn màn hình trắng bốn giây thì đã nghĩ là treo và
> bấm tải lại. Streaming cũng cho phép người dùng ngắt giữa chừng khi thấy
> model đi sai hướng, tiết kiệm cả thời gian lẫn token. Ngược lại, non-streaming
> phù hợp khi đầu ra phải được xử lý trọn vẹn mới dùng được: parse JSON, gọi
> tool, hay chạy lớp kiểm duyệt nội dung trước khi hiển thị, vì không ai muốn
> in nửa câu độc hại rồi mới thu hồi. Nó cũng phù hợp cho tác vụ chạy nền và
> xử lý theo lô, nơi không có người chờ nên độ trễ cảm nhận vô nghĩa, và code
> gọn hơn hẳn vì không phải quản lý vòng lặp chunk cùng trạng thái dở dang.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Với delay cố định, mỗi client vẫn đập vào server cùng một nhịp bất kể server
> đang khỏe hay sắp sập, nên lượng tải trong lúc sự cố gần như không giảm.
> Exponential backoff giãn khoảng chờ gấp đôi sau mỗi lần hỏng, nên tải giảm
> theo cấp số nhân đúng vào lúc server cần khoảng lặng để hồi phục, trong khi
> vẫn thử lại rất nhanh ở lần đầu nếu đó chỉ là trục trặc thoáng qua.
>
> Nếu hàng nghìn client cùng retry với delay cố định giống nhau thì xảy ra
> hiệu ứng đàn sấm: tất cả bị lỗi gần như cùng lúc, nên tất cả cũng thử lại
> cùng lúc sau đúng một giây, tạo ra những đợt sóng tải đồng bộ đập vào server
> đang yếu. Mỗi đợt sóng lại làm thêm nhiều request hỏng, sinh ra đợt sóng kế
> tiếp còn lớn hơn, và hệ thống rơi vào vòng xoáy không tự thoát ra được ngay
> cả khi nguyên nhân gốc đã hết. Bản thân retry trở thành nguồn gây sự cố.
>
> Trong sản phẩm thật thì backoff thôi chưa đủ, cần cộng thêm jitter, tức một
> lượng ngẫu nhiên vào mỗi khoảng chờ, để phá vỡ sự đồng bộ giữa các client.
> Hai lớp bảo vệ thường đi kèm là trần thời gian chờ để không retry sau nửa
> tiếng, và circuit breaker để ngừng hẳn khi tỷ lệ lỗi vượt ngưỡng.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona mình chọn: **"Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn
> gọn bằng tiếng Việt. Nếu không chắc chắn, hãy nói rõ là chưa chắc thay vì
> đoán."**
>
> Cụm **"trả lời ngắn gọn"** không phải chuyện văn phong mà là chuyện kỹ thuật.
> Mỗi lượt trả lời đều bị nhét lại vào `history` và gửi kèm ở lượt sau, nên câu
> trả lời dài làm phình token đầu vào của mọi lượt tiếp theo, chi phí tăng dồn
> chứ không tăng tuyến tính. Thêm nữa `history` chỉ giữ 6 message, nên một câu
> trả lời lan man sẽ chiếm chỗ và đẩy ngữ cảnh hữu ích ra ngoài sớm hơn.
>
> Cụm **"bằng tiếng Việt"** ghim ngôn ngữ đầu ra. Không nói rõ thì model có xu
> hướng trôi sang tiếng Anh khi câu hỏi chứa nhiều thuật ngữ kỹ thuật, và vì
> system prompt được dựng lại ở đầu mỗi lượt nên chỉ thị này còn sống sót kể
> cả sau khi `history` bị cắt. Đó chính là lý do persona phải nằm ngoài
> `history` chứ không nằm trong.
>
> Câu cuối về việc thừa nhận chưa chắc là hàng rào chống bịa đặt. Trợ giảng nói
> sai một khái niệm nền sẽ khiến học viên học lệch mà không biết, nên thà nhận
> không biết còn hơn trả lời trôi chảy mà sai.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là cách cắt `history`. Hàm đang cắt theo **số message**
> bằng `history[-6:]`, nghĩa là 3 lượt gần nhất bất kể mỗi lượt dài bao nhiêu.
> Điều này hỏng theo cả hai chiều. Khi người dùng dán vào một đoạn log dài,
> ba lượt đó vẫn có thể vượt cửa sổ ngữ cảnh và API trả lỗi. Ngược lại khi
> người dùng hỏi những câu ngắn, trợ lý quên mất thông tin từ lượt thứ tư dù
> ngân sách token vẫn còn thừa rất nhiều. Nói cách khác, số message là chỉ báo
> sai cho thứ thật sự bị giới hạn, vốn là token.
>
> Cải thiện cụ thể: **cắt theo ngân sách token thay vì theo số message.** Đặt
> một hằng số, ví dụ `MAX_HISTORY_TOKENS = 2000`. Sau mỗi lượt, duyệt
> `history` từ message mới nhất ngược về đầu, cộng dồn `count_tokens` của từng
> `content`, và dừng ngay trước message đầu tiên làm tổng vượt ngưỡng. Giữ lại
> phần đã duyệt theo đúng thứ tự thời gian. Cần bỏ luôn message user nếu
> message assistant tương ứng bị loại, để hội thoại không còn câu hỏi cụt không
> có câu trả lời. Chi phí triển khai chỉ khoảng mười dòng, và tái dùng đúng
> `count_tokens` đã viết ở Task 2.2 nên không thêm phụ thuộc nào.
>
> Bước tiếp theo nếu có thêm thời gian là tóm tắt cuốn chiếu: khi phần bị cắt
> vượt một ngưỡng nào đó, gọi model tóm tắt nó thành vài câu rồi nhét bản tóm
> tắt vào ngay sau system prompt. Trợ lý giữ được mạch hội thoại dài mà vẫn
> khống chế được token, đổi lại là một lời gọi API phụ mỗi khi cắt.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
