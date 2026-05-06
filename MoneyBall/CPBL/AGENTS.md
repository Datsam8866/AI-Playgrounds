# AGENTS.md — Baseball 專案爬蟲規則

## 爬蟲工具選擇原則

| 情境 | 工具 | 使用方式 |
|---|---|---|
| 社群媒體（PTT、Twitter/X、Facebook 等） | **Playwright MCP** | 直接叫 Codex 操作，或寫 Python 腳本 |
| 一般靜態/半靜態網站 | **Firecrawl MCP** | 直接叫 Codex 爬取，或寫 Python 腳本 |
| CPBL 官網 API（`/box/getlive` 等） | **requests** | 已知 API 端點，不需要瀏覽器 |

## MCP 設定（全域 user 層級）

Playwright 與 Firecrawl 已安裝為 **全域 MCP server**（`~/.Codex.json`），在任何專案皆可直接使用。

```
playwright:  npx @playwright/mcp@latest  ✓ Connected
firecrawl:   npx firecrawl-mcp           ✓ Connected
```

**使用方式**：直接對 Codex 下指令，例如：
- 「用 Firecrawl 爬這個網址，給我 Markdown」
- 「用 Playwright 打開這個頁面，截圖給我」

重新啟動 Codex 後 MCP 即生效，無需額外設定。

---

## Firecrawl

- **API Key**：`fc-0e79b5dee91e44979373f6052b9cd351`
- **MCP**：已設定於全域（含 API Key 環境變數）
- **Python 套件**：`firecrawl-py` v4.22.1（寫腳本時用）

```python
from firecrawl import FirecrawlApp

app = FirecrawlApp(api_key="fc-0e79b5dee91e44979373f6052b9cd351")

# 單頁爬取，回傳 Markdown
result = app.scrape_url("https://...", formats=["markdown"])
print(result.markdown)

# 整站爬取（crawl）
result = app.crawl_url("https://...", limit=50)
```

---

## Playwright

- **MCP**：已設定於全域（`@playwright/mcp@latest`）
- **Python 套件**：`playwright` v1.58.0，Chromium 已安裝（寫腳本時用）
- **用途**：需要登入、需要 JS 渲染、社群媒體

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("https://...")
    # 等待特定元素出現
    page.wait_for_selector("div.content")
    content = page.inner_text("div.content")
    browser.close()
```

### 社群媒體注意事項

- **PTT**：直接用 `requests` 或 Playwright 爬 `https://www.ptt.cc/bbs/Baseball/`，需帶 cookie `over18=1`
- **Twitter/X**：需要登入帳號，用 Playwright 維持 session；或使用官方 API
- **反爬機制**：加 `page.wait_for_timeout(1000)` 模擬人工延遲，避免被封

---

## MCP 管理指令

```powershell
# 查看所有 MCP 狀態
Codex mcp list

# 重新加入（如遺失）
Codex mcp add playwright -s user -- npx @playwright/mcp@latest
Codex mcp add firecrawl -s user -e FIRECRAWL_API_KEY=fc-0e79b5dee91e44979373f6052b9cd351 -- npx firecrawl-mcp
```

## Python 環境檢查

```powershell
python -c "from playwright.sync_api import sync_playwright; print('Playwright OK')"
python -c "from firecrawl import FirecrawlApp; print('Firecrawl OK')"
python -m playwright install chromium  # 若瀏覽器遺失時重裝
```
