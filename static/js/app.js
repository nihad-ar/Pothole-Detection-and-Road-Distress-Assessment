// RoadScan AI - Client Controller (Connected to Input Images/ Dataset)
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

  const pciScoreEl = document.getElementById("pciScore");
  const pciRatingBadge = document.getElementById("pciRatingBadge");
  const potholeCountEl = document.getElementById("potholeCount");
  const potholeAreaText = document.getElementById("potholeAreaText");
  const crackDensityEl = document.getElementById("crackDensity");
  const fatigueStatusText = document.getElementById("fatigueStatusText");
  const actionRecommendationEl = document.getElementById("actionRecommendation");
  const potholeListEl = document.getElementById("potholeList");

  // Stages
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
    valClahe.textContent = e.target.value;
    state.clahe_clip = parseFloat(e.target.value);
  });
  paramClahe.addEventListener("change", () => runInspection());

  paramGamma.addEventListener("input", (e) => {
    valGamma.textContent = e.target.value;
    state.gamma = parseFloat(e.target.value);
  });
  paramGamma.addEventListener("change", () => runInspection());

  paramCanny.addEventListener("input", (e) => {
    const val = parseInt(e.target.value);
    valCanny.textContent = `${val} - ${val * 3}`;
    state.canny1 = val;
  });
  paramCanny.addEventListener("change", () => runInspection());

  paramArea.addEventListener("input", (e) => {
    valArea.textContent = e.target.value;
    state.min_area = parseInt(e.target.value);
  });
  paramArea.addEventListener("change", () => runInspection());

  // Rescan Folder Button
  rescanFolderBtn.addEventListener("click", () => {
    rescanFolderBtn.classList.add("animate-pulse");
    loadDatasetImages().finally(() => {
      rescanFolderBtn.classList.remove("animate-pulse");
    });
  });

  // Load and render images from Input Images folder
  async function loadDatasetImages() {
    try {
      const res = await fetch("/api/samples");
      const data = await res.json();
      state.totalCount = data.total_count || 0;

      datasetSourceBadge.textContent = `Folder: Input Images/ (${state.totalCount.toLocaleString()} images)`;

      // Render Featured Pills
      featuredPillsContainer.innerHTML = "";
      const featured = data.featured || [];

      featured.forEach((img, idx) => {
        const btn = document.createElement("button");
        const isActive = (state.currentSample === img.filename) || (!state.currentSample && idx === 0);
        btn.className = isActive
          ? "sample-pill active px-3 py-1.5 text-xs rounded-xl border border-amber-500/40 bg-amber-500/10 text-amber-300 flex items-center gap-1.5 transition font-semibold"
          : "sample-pill px-3 py-1.5 text-xs rounded-xl border border-slate-800 bg-slate-800/60 text-slate-300 hover:bg-slate-800 flex items-center gap-1.5 transition";
        btn.dataset.sample = img.filename;
        btn.innerHTML = `<span>🇮🇳</span> <span>${img.label}</span>`;

        btn.addEventListener("click", () => {
          document.querySelectorAll(".sample-pill").forEach(p => {
            p.className = "sample-pill px-3 py-1.5 text-xs rounded-xl border border-slate-800 bg-slate-800/60 text-slate-300 hover:bg-slate-800 flex items-center gap-1.5 transition";
          });
          btn.className = "sample-pill active px-3 py-1.5 text-xs rounded-xl border border-amber-500/40 bg-amber-500/10 text-amber-300 flex items-center gap-1.5 transition font-semibold";
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
      datasetDropdown.innerHTML = `<option value="">Select from Dataset (${(data.dataset_sample || []).length} available)...</option>`;
      const datasetSamples = data.dataset_sample || [];
      datasetSamples.forEach(img => {
        const opt = document.createElement("option");
        opt.value = img.filename;
        opt.textContent = img.filename;
        datasetDropdown.appendChild(opt);
      });

      datasetDropdown.addEventListener("change", (e) => {
        if (!e.target.value) return;
        document.querySelectorAll(".sample-pill").forEach(p => {
          p.className = "sample-pill px-3 py-1.5 text-xs rounded-xl border border-slate-800 bg-slate-800/60 text-slate-300 hover:bg-slate-800 flex items-center gap-1.5 transition";
        });
        state.currentSample = e.target.value;
        state.customImageData = null;
        runInspection();
      });

      if (state.currentSample) {
        runInspection();
      }
    } catch (err) {
      console.error("Failed to load dataset images:", err);
      datasetSourceBadge.textContent = "Folder: Input Images/ (Error loading)";
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
      document.querySelectorAll(".sample-pill").forEach(p => {
        p.className = "sample-pill px-3 py-1.5 text-xs rounded-xl border border-slate-800 bg-slate-800/60 text-slate-300 hover:bg-slate-800 flex items-center gap-1.5 transition";
      });
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
      modal.className = "fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4";
      modal.innerHTML = `
        <div class="bg-slate-900 p-6 rounded-2xl border border-slate-800 space-y-4 max-w-lg w-full">
          <h3 class="text-sm font-bold text-white flex items-center gap-2">
            <i data-lucide="camera" class="w-4 h-4 text-amber-400"></i> Capture Road Pavement Snapshot
          </h3>
          <div id="videoContainer" class="aspect-video bg-black rounded-xl overflow-hidden flex items-center justify-center"></div>
          <div class="flex justify-end gap-2">
            <button id="cancelWebcam" class="px-4 py-2 bg-slate-800 text-slate-300 text-xs rounded-xl hover:bg-slate-700">Cancel</button>
            <button id="snapWebcam" class="px-4 py-2 bg-amber-600 text-white text-xs font-semibold rounded-xl hover:bg-amber-500">Capture & Inspect</button>
          </div>
        </div>
      `;
      document.body.appendChild(modal);
      modal.querySelector("#videoContainer").appendChild(video);
      if (window.lucide) window.lucide.createIcons();

      modal.querySelector("#cancelWebcam").addEventListener("click", () => {
        stream.getTracks().forEach(t => t.stop());
        modal.remove();
      });

      modal.querySelector("#snapWebcam").addEventListener("click", () => {
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth || 800;
        canvas.height = video.videoHeight || 600;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        state.customImageData = canvas.toDataURL("image/jpeg");
        state.currentSample = "webcam_snapshot.jpg";
        stream.getTracks().forEach(t => t.stop());
        modal.remove();
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
        alert("Inspection Error: " + data.error);
        return;
      }

      currentFileDisplay.textContent = `Analyzing: ${data.active_filename || state.currentSample}`;

      // Update Main Images
      rawRoadImg.src = data.raw_image;
      annotatedRoadImg.src = data.annotated_image;

      // Update Metrics
      const m = data.metrics;
      pciScoreEl.textContent = m.pci_score;

      // Badge Styling
      pciRatingBadge.textContent = m.condition_rating;
      if (m.pci_score >= 88) {
        pciRatingBadge.className = "inline-block mt-2 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20";
        pciScoreEl.className = "text-3xl font-extrabold text-emerald-400";
      } else if (m.pci_score >= 68) {
        pciRatingBadge.className = "inline-block mt-2 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-teal-500/10 text-teal-400 border border-teal-500/20";
        pciScoreEl.className = "text-3xl font-extrabold text-teal-400";
      } else if (m.pci_score >= 42) {
        pciRatingBadge.className = "inline-block mt-2 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-amber-500/10 text-amber-400 border border-amber-500/20";
        pciScoreEl.className = "text-3xl font-extrabold text-amber-400";
      } else {
        pciRatingBadge.className = "inline-block mt-2 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-rose-500/10 text-rose-400 border border-rose-500/20";
        pciScoreEl.className = "text-3xl font-extrabold text-rose-400";
      }

      potholeCountEl.textContent = m.pothole_count;
      potholeAreaText.textContent = `Area: ${m.total_pothole_area_px.toLocaleString()} px² (${m.pothole_coverage_pct}%)`;

      crackDensityEl.textContent = `${m.crack_density_pct} %`;
      fatigueStatusText.textContent = m.is_fatigue_alligator ? "Fatigue: Alligator Cracking" : "Fatigue: Normal / Intact";
      fatigueStatusText.className = m.is_fatigue_alligator ? "text-xs text-rose-400 font-semibold mt-1" : "text-xs text-slate-400 mt-1";

      actionRecommendationEl.textContent = m.action_recommendation;

      // Update Potholes List
      potholeListEl.innerHTML = "";
      if (m.potholes && m.potholes.length > 0) {
        m.potholes.forEach(p => {
          const div = document.createElement("div");
          div.className = "bg-slate-950 p-2 rounded-xl border border-slate-800/80 flex items-center justify-between";
          div.innerHTML = `
            <div>
              <span class="font-bold text-white">#${p.id} [${p.severity}]</span>
              <span class="text-[10px] text-slate-400 block">${p.approx_size_cm}</span>
            </div>
            <span class="font-mono text-amber-400 font-semibold">${p.area_px.toLocaleString()} px²</span>
          `;
          potholeListEl.appendChild(div);
        });
      } else {
        potholeListEl.innerHTML = `<div class="p-4 text-center text-slate-500 italic text-xs">No deep pothole cavities detected on this pavement surface.</div>`;
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
      alert("Inspection Request Failed: " + err.message);
    } finally {
      scanLoading.classList.add("hidden");
    }
  }

  // Initial Boot
  loadDatasetImages();
});
