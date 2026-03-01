const recordBtn = document.getElementById("recordBtn");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
// const API_BASE = "http://127.0.0.1:8000";
const API_BASE = "http://88.126.86.197:16384";

let mediaRecorder;
let chunks = [];

recordBtn.addEventListener("click", async () => {
    resultEl.textContent = "";
    statusEl.textContent = "Requesting microphone permission...";

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        statusEl.textContent = "Recording for 5 seconds...";
        chunks = [];

        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) {
                chunks.push(e.data);
            }
        };

        mediaRecorder.onstop = async () => {
            statusEl.textContent = "Processing audio...";

            const blob = new Blob(chunks, { type: "audio/webm" });
            const formData = new FormData();
            // Name "file" must match the FastAPI parameter name
            formData.append("file", blob, "recording.webm");

            try {
                const response = await fetch(`${API_BASE}/predict`, {
                    method: "POST",
                    body: formData
                });

                if (!response.ok) {
                    const errorText = await response.text();
                    statusEl.textContent = "Error from API";
                    resultEl.textContent = errorText;
                    return;
                }

                const data = await response.json();
                statusEl.textContent = "Done.";
                resultEl.textContent =
                    `Language: ${data.label ?? "unknown"} (index=${data.predicted_index}, confidence=${data.confidence.toFixed(3)})`;
            } catch (err) {
                statusEl.textContent = "Network error";
                resultEl.textContent = err.toString();
            }
        };

        mediaRecorder.start();

        // Stop after 5 seconds
        setTimeout(() => {
            mediaRecorder.stop();
            stream.getTracks().forEach(track => track.stop());
        }, 5000);
    } catch (err) {
        statusEl.textContent = "Microphone permission denied or error.";
        resultEl.textContent = err.toString();
    }
});
