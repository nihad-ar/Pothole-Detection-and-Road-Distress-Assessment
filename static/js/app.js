// RoadScan AI - Professional Engineering Client Controller
document.addEventListener("DOMContentLoaded", () => {
  if (window.lucide) {
    window.lucide.createIcons();
  }

  // State
  const state = {
    currentSample: "",
    customImageData: null,
    gamma: 0.7,
    clahe_clip: 2.0,
    canny1: 60,
    min_area: 1500,
    totalCount: 0
  };

  // DOM Elements
  const rawRoadImg = document.getElementById("rawRoadImg");
  const annotatedRoadImg = document.getElementById("annotatedRoadImg");
  const scanLoading = document.getElementById("scanLoading");
  const featuredPillsContainer = document.getElementById("featuredPillsContainer");
  const datasetDropdown = document.getElementById("datasetDropdown");
  const currentFileDisplay = document.getElementById("currentFileDisplay");
  const datasetSourceBadge = document.getElementById("datasetSourceBadge");
  const rescanFolderBtn = document.getElementById("rescanFolderBtn");

  // Metrics
  const pciScoreEl = document.getElementById("pciScore");
  const pciRatingBadge = document.getElementById("pciRatingBadge");
  const pciSubStatus = document.getElementById("pciSubStatus");

  const potholeCountEl = document.getElementById("potholeCount");
  const potholeAreaText = document.getElementById("potholeAreaText");
  const potholePctText = document.getElementById("potholePctText");

  const crackDensityEl = document.getElementById("crackDensity");
  const fatigueStatusText = document.getElementById("fatigueStatusText");

  const actionHeadline = document.getElementById("actionHeadline");
  const actionPriority = document.getElementById("actionPriority");
  const potholeListEl = document.getElementById("potholeList");

  // Pipeline Stages
  const stage1Img = document.getElementById("stage1Img");
  const stage2Img = document.getElementById("stage2Img");
  const stage3Img = document.getElementById("stage3Img");
  const stage4Img = document.getElementById("stage4Img");
  const stage5Img = document.getElementById("stage5Img");
  const stage6Img = document.getElementById("stage6Img");

  // Sliders
  const paramClahe = document.getElementById("param_clahe");
  const paramGamma = document.getElementById("param_gamma");
  const paramCanny = document.getElementById("param_canny");
  const paramArea = document.getElementById("param_area");

  const valClahe = document.getElementById("val_clahe");
  const valGamma = document.getElementById("val_gamma");
  const valCanny = document.getElementById("val_canny");
  const valArea = document.getElementById("val_area");

  // Slider event listeners
  paramClahe.addEventListener("input", (e) => {
    valClahe.textContent = parseFloat(e.target.value).toFixed(1);
    state.clahe_clip = parseFloat(e.target.value);
  });
  paramClahe.addEventListener("change", () => runInspection());

  paramGamma.addEventListener("input", (e) => {
    valGamma.textContent = parseFloat(e.target.value).toFixed(2);
    state.gamma = parseFloat(e.target.value);
  });
  paramGamma.addEventListener("change", () => runInspection());

  paramCanny.addEventListener("input", (e) => {
    const val = parseInt(e.target.value, 10);
    valCanny.textContent = `${val} - ${val * 3}`;
    state.canny1 = val;
  });
  paramCanny.addEventListener("change", () => runInspection());

  paramArea.addEventListener("input", (e) => {
    valArea.textContent = `${parseInt(e.target.value, 10)} px`;
    state.min_area = parseInt(e.target.value, 10);
  });
  paramArea.addEventListener("change", () => runInspection());

  // Rescan Folder Button
  rescanFolderBtn.addEventListener("click", () => {
    rescanFolderBtn.style.opacity = "0.5";
    loadDatasetImages().finally(() => {
      rescanFolderBtn.style.opacity = "1";
    });
  });

  // Helper: Shorten sample pill labels for clean, non-overflowing UI
  function formatSampleLabel(filename) {
    const lower = filename.toLowerCase();
    if (lower.includes("delhi")) return "New Delhi (Severe Potholes)";
    if (lower.includes("bengaluru") || lower.includes("munnekollala")) return "Bengaluru (Fatigue Cracks)";
    let clean = filename.replace(/\.[^/.]+$/, "").replace(/_/g, " ").replace(/-/g, " ");
    return clean.length > 25 ? clean.substring(0, 22) + "..." : clean;
  }

  // Load and render images from Input Images folder
  async function loadDatasetImages() {
    try {
      const res = await fetch("/api/samples");
      const data = await res.json();
      state.totalCount = data.total_count || 0;

      datasetSourceBadge.textContent = `Input Images/ (${state.totalCount.toLocaleString()} photos)`;

      // Render Featured Pills
      featuredPillsContainer.innerHTML = "";
      const featured = data.featured || [];

      featured.forEach((img, idx) => {
        const btn = document.createElement("button");
        const isActive = (state.currentSample === img.filename) || (!state.currentSample && idx === 0);
        btn.className = isActive ? "pill-item active" : "pill-item";
        btn.dataset.sample = img.filename;
        btn.textContent = formatSampleLabel(img.filename);

        btn.addEventListener("click", () => {
          document.querySelectorAll(".pill-item").forEach(p => p.classList.remove("active"));
          btn.classList.add("active");
          datasetDropdown.value = "";

          state.currentSample = img.filename;
          state.customImageData = null;
          runInspection();
        });

        featuredPillsContainer.appendChild(btn);

        if (isActive && !state.currentSample) {
          state.currentSample = img.filename;
        }
      });

      // Render Dataset Dropdown
      datasetDropdown.innerHTML = `<option value="">Select from dataset (${(data.dataset_sample || []).length} available)...</option>`;
      const datasetSamples = data.dataset_sample || [];
      datasetSamples.forEach(img => {
        const opt = document.createElement("option");
        opt.value = img.filename;
        opt.textContent = img.filename.length > 30 ? img.filename.substring(0, 28) + "..." : img.filename;
        datasetDropdown.appendChild(opt);
      });

      datasetDropdown.addEventListener("change", (e) => {
        if (!e.target.value) return;
        document.querySelectorAll(".pill-item").forEach(p => p.classList.remove("active"));
        state.currentSample = e.target.value;
        state.customImageData = null;
        runInspection();
      });

      if (state.currentSample) {
        runInspection();
      }
    } catch (err) {
      console.error("Failed to load dataset images:", err);
      datasetSourceBadge.textContent = "Input Images/ (Offline)";
    }
  }

  // Re-run scan button
  document.getElementById("runScanBtn").addEventListener("click", () => runInspection());

  // Download diagnostic image
  document.getElementById("downloadAnnotatedBtn").addEventListener("click", () => {
    if (!annotatedRoadImg.src) return;
    const a = document.createElement("a");
    a.href = annotatedRoadImg.src;
    const cleanName = (state.currentSample || "custom").replace(/[^a-zA-Z0-9_-]/g, "_");
    a.download = `roadscan_diagnostic_${cleanName}.jpg`;
    a.click();
  });

  // Upload custom image
  const uploadInput = document.getElementById("imageUpload");
  uploadInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      state.customImageData = event.target.result;
      state.currentSample = file.name;
      document.querySelectorAll(".pill-item").forEach(p => p.classList.remove("active"));
      runInspection();
    };
    reader.readAsDataURL(file);
  });

  // Webcam Capture
  const webcamBtn = document.getElementById("webcamBtn");
  webcamBtn.addEventListener("click", async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      const video = document.createElement("video");
      video.srcObject = stream;
      video.play();

      const modal = document.createElement("div");
      modal.className = "fixed inset-0 bg-[#070a10]/80 backdrop-blur-sm z-50 flex items-center justify-center p-4";
      modal.innerHTML = `
        <div class="panel-card p-6 space-y-4 max-w-lg w-full border border-slate-700">
          <div class="flex items-center justify-between">
            <h3 class="text-sm font-semibold text-slate-100 flex items-center gap-2">
              <i data-lucide="camera" class="w-4 h-4 text-amber-400"></i>
              <span>Capture Road Pavement Photo</span>
            </h3>
            <button id="closeWebcamX" class="text-slate-400 hover:text-slate-200">
              <i data-lucide="x" class="w-4 h-4"></i>
            </button>
          </div>
          <div id="videoContainer" class="aspect-video bg-[#070a10] rounded-lg overflow-hidden flex items-center justify-center border border-slate-700"></div>
          <div class="flex justify-end gap-2 pt-2">
            <button id="cancelWebcam" class="btn-subtle">Cancel</button>
            <button id="snapWebcam" class="btn-amber">Capture & Inspect</button>
          </div>
        </div>
      `;
      document.body.appendChild(modal);
      modal.querySelector("#videoContainer").appendChild(video);
      if (window.lucide) window.lucide.createIcons();

      const cleanup = () => {
        stream.getTracks().forEach(t => t.stop());
        modal.remove();
      };

      modal.querySelector("#cancelWebcam").addEventListener("click", cleanup);
      modal.querySelector("#closeWebcamX").addEventListener("click", cleanup);

      modal.querySelector("#snapWebcam").addEventListener("click", () => {
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth || 800;
        canvas.height = video.videoHeight || 600;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        state.customImageData = canvas.toDataURL("image/jpeg");
        state.currentSample = "webcam_snapshot.jpg";
        cleanup();
        runInspection();
      });
    } catch (err) {
      alert("Unable to access camera: " + err.message);
    }
  });

  // Main API Call
  async function runInspection() {
    scanLoading.classList.remove("hidden");

    const payload = {
      sample_name: state.customImageData ? null : state.currentSample,
      image: state.customImageData || null,
      gamma: state.gamma,
      clahe_clip: state.clahe_clip,
      canny1: state.canny1,
      canny2: state.canny1 * 3,
      min_pothole_area: state.min_area
    };

    try {
      const res = await fetch("/api/inspect_road", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();

      if (data.error) {
        alert("Inspection error: " + data.error);
        return;
      }

      currentFileDisplay.textContent = `Analyzing: ${data.active_filename || state.currentSample}`;

      // Update Main Images
      rawRoadImg.src = data.raw_image;
      annotatedRoadImg.src = data.annotated_image;

      // Update Metrics
      const m = data.metrics;
      pciScoreEl.textContent = m.pci_score;

      // Rating badge styling with punchy status classes
      if (m.pci_score >= 85) {
        pciRatingBadge.className = "tech-tag status-good";
        pciRatingBadge.textContent = "Good";
        pciSubStatus.textContent = "Minimal Distress";
        pciSubStatus.className = "font-medium text-emerald-400 font-mono";
      } else if (m.pci_score >= 60) {
        pciRatingBadge.className = "tech-tag status-warning";
        pciRatingBadge.textContent = "Fair";
        pciSubStatus.textContent = "Moderate Wear";
        pciSubStatus.className = "font-medium text-amber-400 font-mono";
      } else if (m.pci_score >= 40) {
        pciRatingBadge.className = "tech-tag status-warning";
        pciRatingBadge.textContent = "Poor";
        pciSubStatus.textContent = "Structural Cracking";
        pciSubStatus.className = "font-medium text-amber-500 font-mono";
      } else {
        pciRatingBadge.className = "tech-tag status-danger";
        pciRatingBadge.textContent = "Critical";
        pciSubStatus.textContent = "Immediate Repair Required";
        pciSubStatus.className = "font-medium text-rose-400 font-mono";
      }

      potholeCountEl.textContent = m.pothole_count;
      potholeAreaText.textContent = `Area: ${m.total_pothole_area_px.toLocaleString()} px²`;
      potholePctText.textContent = `${m.pothole_coverage_pct}% surface`;

      crackDensityEl.textContent = `${m.crack_density_pct} %`;
      if (m.is_fatigue_alligator) {
        fatigueStatusText.textContent = "Alligator Fatigue Detected";
        fatigueStatusText.className = "text-[11px] text-rose-400 font-semibold";
      } else {
        fatigueStatusText.textContent = "Surface Sound / Intact";
        fatigueStatusText.className = "text-[11px] text-emerald-400 font-medium";
      }

      // Action Work Order Headline
      if (m.pothole_count > 0) {
        actionHeadline.textContent = "Emergency Asphalt Patching";
        actionPriority.textContent = "Priority: CRITICAL";
        actionPriority.className = "font-mono text-rose-400 font-semibold";
      } else if (m.is_fatigue_alligator) {
        actionHeadline.textContent = "Mill & Resurface Overlay";
        actionPriority.textContent = "Priority: HIGH";
        actionPriority.className = "font-mono text-amber-400 font-semibold";
      } else if (m.pci_score < 80) {
        actionHeadline.textContent = "Crack Sealing & Joint Repair";
        actionPriority.textContent = "Priority: MEDIUM";
        actionPriority.className = "font-mono text-amber-300 font-semibold";
      } else {
        actionHeadline.textContent = "Routine Inspection / Monitor";
        actionPriority.textContent = "Priority: LOW";
        actionPriority.className = "font-mono text-emerald-400 font-semibold";
      }

      // Update Potholes List
      potholeListEl.innerHTML = "";
      if (m.potholes && m.potholes.length > 0) {
        m.potholes.forEach(p => {
          const div = document.createElement("div");
          div.className = "panel-raised p-2 flex items-center justify-between text-xs border border-slate-700/60";
          const sevColor = p.severity === "Severe" ? "text-rose-400" : "text-amber-400";
          div.innerHTML = `
            <div>
              <span class="font-semibold text-slate-100">Pothole #${p.id}</span>
              <span class="text-[11px] text-slate-400 block font-mono mt-0.5">${p.approx_size_cm} · <span class="${sevColor} font-medium">${p.severity}</span></span>
            </div>
            <span class="font-mono tabular-nums text-xs text-amber-400 font-semibold">${p.area_px.toLocaleString()} px²</span>
          `;
          potholeListEl.appendChild(div);
        });
      } else {
        potholeListEl.innerHTML = `<div class="p-4 text-center text-slate-400 text-xs">No pothole cavities detected on this pavement surface</div>`;
      }

      // Update 6-Stage Intermediate Inspector
      if (data.stages) {
        stage1Img.src = data.stages.clahe_enhanced;
        stage2Img.src = data.stages.denoised_median;
        stage3Img.src = data.stages.sobel_gradients;
        stage4Img.src = data.stages.crack_network;
        stage5Img.src = data.stages.lbp_texture;
        stage6Img.src = data.stages.pothole_mask;
      }

    } catch (err) {
      console.error(err);
      alert("Inspection request failed: " + err.message);
    } finally {
      scanLoading.classList.add("hidden");
    }
  }

  // Initial Boot
  loadDatasetImages();
});
