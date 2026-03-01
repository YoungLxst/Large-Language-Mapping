from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Language Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en dev seulement
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve static
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <title>Language Detection</title>
        <style>
            :root {
                --bg: #0f172a;
                --card-bg: #020617;
                --accent: #3b82f6;
                --accent-hover: #2563eb;
                --text: #e5e7eb;
                --subtle: #9ca3af;
                --danger: #f97373;
                --success: #4ade80;
                --shadow: 0 20px 40px rgba(15, 23, 42, 0.8);
                --radius: 18px;
            }

            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }

            body {
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                background: radial-gradient(circle at top left, #1f2937, #020617);
                color: var(--text);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 24px;
            }

            .card {
                background: radial-gradient(circle at top, rgba(59, 130, 246, 0.15), transparent 55%) var(--card-bg);
                border-radius: var(--radius);
                box-shadow: var(--shadow);
                padding: 28px 32px 26px;
                max-width: 480px;
                width: 100%;
                border: 1px solid rgba(148, 163, 184, 0.22);
                position: relative;
                overflow: hidden;
            }

            .card::before {
                content: "";
                position: absolute;
                inset: 0;
                background: radial-gradient(circle at 0% 0%, rgba(59, 130, 246, 0.18), transparent 55%);
                opacity: 0.9;
                pointer-events: none;
            }

            .card-inner {
                position: relative;
                z-index: 1;
            }

            h1 {
                font-size: 1.6rem;
                font-weight: 650;
                letter-spacing: 0.02em;
                margin-bottom: 4px;
            }

            .subtitle {
                font-size: 0.9rem;
                color: var(--subtle);
                margin-bottom: 22px;
            }

            .pill {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                font-size: 0.75rem;
                padding: 4px 9px;
                border-radius: 999px;
                background: rgba(15, 118, 110, 0.1);
                color: #6ee7b7;
                border: 1px solid rgba(16, 185, 129, 0.35);
                margin-bottom: 12px;
            }

            .pill-dot {
                width: 7px;
                height: 7px;
                border-radius: 999px;
                background: #22c55e;
                box-shadow: 0 0 8px rgba(34, 197, 94, 0.9);
            }

            .controls {
                display: flex;
                align-items: center;
                gap: 12px;
                margin-bottom: 14px;
            }

            #recordBtn {
                appearance: none;
                border: none;
                cursor: pointer;
                border-radius: 999px;
                padding: 10px 18px;
                font-size: 0.95rem;
                font-weight: 580;
                letter-spacing: 0.01em;
                background: radial-gradient(circle at 30% 0, #60a5fa, #2563eb);
                color: white;
                display: inline-flex;
                align-items: center;
                gap: 8px;
                box-shadow: 0 10px 25px rgba(37, 99, 235, 0.5);
                transition: transform 0.12s ease-out, box-shadow 0.12s ease-out, background 0.15s ease-out;
            }

            #recordBtn::before {
                content: "";
                width: 10px;
                height: 10px;
                border-radius: 999px;
                background: #ef4444;
                box-shadow: 0 0 10px rgba(239, 68, 68, 0.85);
            }

            #recordBtn:hover {
                transform: translateY(-1px) scale(1.01);
                box-shadow: 0 14px 28px rgba(37, 99, 235, 0.6);
                background: radial-gradient(circle at 30% 0, #93c5fd, #1d4ed8);
            }

            #recordBtn:active {
                transform: translateY(0) scale(0.99);
                box-shadow: 0 6px 14px rgba(37, 99, 235, 0.7);
            }

            .hint {
                font-size: 0.8rem;
                color: var(--subtle);
            }

            .status-label {
                font-size: 0.8rem;
                font-weight: 550;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: var(--subtle);
                margin-top: 4px;
                margin-bottom: 4px;
            }

            #status {
                font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
                font-size: 0.85rem;
                min-height: 1.2em;
                color: #e5e7eb;
            }

            #status.ok {
                color: var(--success);
            }

            #status.error {
                color: var(--danger);
            }

            #result {
                margin-top: 10px;
                padding: 8px 10px;
                border-radius: 10px;
                background: rgba(15, 23, 42, 0.85);
                border: 1px solid rgba(148, 163, 184, 0.35);
                font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
                font-size: 0.85rem;
                color: #e5e7eb;
                min-height: 1.2em;
                white-space: pre-wrap;
                word-break: break-word;
            }

            .footer {
                margin-top: 14px;
                font-size: 0.78rem;
                color: var(--subtle);
                display: flex;
                justify-content: space-between;
                gap: 10px;
                flex-wrap: wrap;
            }

            .footer span {
                opacity: 0.95;
            }

            @media (max-width: 520px) {
                .card {
                    padding: 22px 18px 20px;
                }

                h1 {
                    font-size: 1.35rem;
                }

                .controls {
                    flex-direction: column;
                    align-items: flex-start;
                }

                #recordBtn {
                    width: 100%;
                    justify-content: center;
                }
            }
        </style>
    </head>
    <body>
        <main class="card">
            <div class="card-inner">
                <div class="pill">
                    <span class="pill-dot"></span>
                    <span>Live language detection</span>
                </div>
                <h1>Language Detection</h1>
                <p class="subtitle">Record 5 seconds of audio and detect the spoken language directly in your browser.</p>

                <div class="controls">
                    <button id="recordBtn">Record 5 seconds</button>
                    <span class="hint">You might be prompted to allow microphone access.</span>
                </div>

                <p class="status-label">Status</p>
                <p id="status"></p>

                <p class="status-label" style="margin-top: 10px;">Result</p>
                <p id="result"></p>

                <div class="footer">
                    <span>Functional demo</span>
                    <span>Sample length: ~5 seconds</span>
                </div>
            </div>
        </main>
        <script src="/static/recorder.js">
        console.log(API_BASE)
        </script>
    </body>
    </html>
    """
