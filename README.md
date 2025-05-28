# 🧠 Bias Buster

**Bias Buster** is an AI-powered tool that analyzes news articles for misinformation, manipulative language, and bias. Whether you're reading political commentary, opinion pieces, or breaking news — this app helps you stay sharp and think critically.

> Built with 💻 Flask + 🧠 Llama 4 via Groq API  
> Frontend styled with Tailwind CSS, fully local & lightweight.

---

## ⚡ Features

- 📰 **URL or Text Input** — Analyze articles from links or paste text directly
- 🧠 **AI-Powered Analysis** — Detects sentiment, tone, theme, and bias patterns
- 🚩 **Highlights Misinformation** — Flags techniques like *generalization*, *ad hominem*, *red herrings*, and more
- 🔍 **Tooltips with Explanations** — Every pattern includes detailed reasoning and trusted source suggestions
- 🤝 **Compare Mode** — Analyze and compare two articles side-by-side

---

## 🛠️ How to Run Locally

1. **Clone the repo**
```bash
git clone https://github.com/your-username/bias-buster.git
cd bias-buster
```
2. **Install dependencies**
pip install -r requirements.txt

3. **Replace Groq API key**
Replace groq api key at line 10:
```10: GROQ_API_KEY = "Your Key"```

4. Run app and go to http://127.0.0.1:5000
