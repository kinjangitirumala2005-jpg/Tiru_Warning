// Early-Warning Wildfire Smoke Detector Frontend JS

document.addEventListener("DOMContentLoaded", () => {
    initDropzones();
    initStatsPolling();
});

function initDropzones() {
    const dropzone = document.getElementById("upload-dropzone");
    const fileInput = document.getElementById("file-input");
    const previewContainer = document.getElementById("preview-container");
    const previewImage = document.getElementById("preview-image");
    const uploadForm = document.getElementById("upload-form");

    if (!dropzone || !fileInput) return;

    ["dragenter", "dragover"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("dragover");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("dragover");
        }, false);
    });

    dropzone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            fileInput.files = files;
            showPreview(files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (fileInput.files.length > 0) {
            showPreview(fileInput.files[0]);
        }
    });

    function showPreview(file) {
        if (file && file.type.startsWith("image/")) {
            const reader = new FileReader();
            reader.onload = (e) => {
                if (previewImage) {
                    previewImage.src = e.target.result;
                    previewContainer?.classList.remove("d-none");
                }
            };
            reader.readAsDataURL(file);
        }
    }
}

function initStatsPolling() {
    const statCards = document.getElementById("stats-section");
    if (!statCards) return;

    // Refresh dashboard stats every 8 seconds
    setInterval(async () => {
        try {
            const res = await fetch("/api/stats");
            if (!res.ok) return;
            const data = await res.json();
            
            const totalEl = document.getElementById("stat-total");
            const smokeEl = document.getElementById("stat-smoke");
            const clearEl = document.getElementById("stat-clear");
            const alertsEl = document.getElementById("stat-alerts");

            if (totalEl) totalEl.textContent = data.total_detections;
            if (smokeEl) smokeEl.textContent = data.total_smoke;
            if (clearEl) clearEl.textContent = data.total_clear;
            if (alertsEl) alertsEl.textContent = data.total_alerts;
        } catch (err) {
            console.debug("Stats polling pause:", err);
        }
    }, 8000);
}

// Test SMTP Alert button trigger
async function triggerTestAlert() {
    const btn = document.getElementById("btn-test-alert");
    const statusMsg = document.getElementById("test-alert-status");
    if (btn) btn.disabled = true;

    try {
        if (statusMsg) statusMsg.innerHTML = `<span class="text-info"><i class="bi bi-hourglass-split"></i> Sending test notification...</span>`;
        const res = await fetch("/api/test-alert", { method: "POST" });
        const data = await res.json();

        if (data.success) {
            if (statusMsg) statusMsg.innerHTML = `<span class="text-success"><i class="bi bi-check-circle-fill"></i> ${data.message}</span>`;
        } else {
            if (statusMsg) statusMsg.innerHTML = `<span class="text-danger"><i class="bi bi-exclamation-triangle-fill"></i> Error: ${data.message}</span>`;
        }
    } catch (err) {
        if (statusMsg) statusMsg.innerHTML = `<span class="text-danger">Request failed: ${err}</span>`;
    } finally {
        if (btn) btn.disabled = false;
    }
}

