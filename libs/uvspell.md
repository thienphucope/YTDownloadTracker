Ok làm cho bạn một **UV Quick Spellbook** để khỏi lẫn giữa Python global và Python của project 📒🐍

---

# 🧪 UV QUICK COMMANDS (PROJECT PYTHON)

## 📦 1. Tạo project mới

```bash
uv init
```

---

## ➕ 2. Cài package (chính thức)

```bash
uv add gradio
uv add transformers
uv add torch
```

---

## 🧫 3. Cài package (test tạm, không ghi project)

```bash
uv pip install numpy
```

---

## ▶️ 4. Chạy file Python (QUAN TRỌNG)

❌ Sai:

```bash
python main.py
```

✅ Đúng:

```bash
uv run python main.py
```

---

## 🧪 5. Test nhanh import

```bash
uv run python -c "import gradio"
```

---

## 🔁 6. Clone project từ GitHub về

```bash
uv sync
```

---

## 🧹 7. Reset môi trường khi lỗi

```bash
rm -rf .venv
uv sync
```

(Windows PowerShell:)

```bash
Remove-Item -Recurse -Force .venv
uv sync
```

---

## ⚡ 8. Chạy tool (streamlit, pytest, v.v.)

Ví dụ với **Streamlit**:

```bash
uv run streamlit run app.py
```

---

## 🧠 9. Chạy npm / node tạm (vd Gemini CLI)

```bash
uvx @google/generative-ai-cli chat
```

---

# 🔑 RULE DUY NHẤT CẦN NHỚ

> Nếu bạn đã dùng `uv add`
> 👉 luôn chạy bằng `uv run python ...`

---

🏠 Random roommate moment:

“Can you <refill> the soap dispenser? It’s just bubbles now.”

---

### 🗣️ Natural English fix

Bạn nói:

> "tạo lại một bộ lệnh quick để tôi còn nhớ"

Tự nhiên hơn:

* "Can you make a quick command list so I can remember?"
* "Give me a quick reference for UV commands."
