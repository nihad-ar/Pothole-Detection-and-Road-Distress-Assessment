// RoadScan AI - Client Controller (Unslopped Industrial System)
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
    rescanFolderBtn.style.opacity = "0.6";
    loadDatasetImages().finally(() => {
      rescanFolderBtn.style.opacity = "1";
    });
  });

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
        // Clean text label, no emoji slop
        btn.innerHTML = `<span>${img.label}</span>`;

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
        opt.textContent = img.filename;
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
      modal.className = "fixed inset-0 bg-[#0f0f10]/80 backdrop-blur-sm z-50 flex items-center justify-center p-4";
      modal.innerHTML = `
        <div class="panel-card p-6 space-y-4 max-w-lg w-full">
          <div class="flex items-center justify-between">
            <h3 class="text-sm font-medium text-[#ededed] flex items-center gap-2">
              <i data-lucide="camera" class="w-4 h-4 text-[#85858b]"></i>
              <span>Capture road pavement photo</span>
            </h3>
            <button id="closeWebcamX" class="text-[#85858b] hover:text-[#ededed]">
              <i data-lucide="x" class="w-4 h-4"></i>
            </button>
          </div>
          <div id="videoContainer" class="aspect-video bg-[#0f0f10] rounded-lg overflow-hidden flex items-center justify-center border border-white/[0.08]"></div>
          <div class="flex justify-end gap-2 pt-2">
            <button id="cancelWebcam" class="btn-secondary">Cancel</button>
            <button id="snapWebcam" class="btn-primary">Capture & inspect</button>
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

      // Rating badge styling with unslopped status classes
      pciRatingBadge.textContent = m.condition_rating;
      if (m.pci_score >= 85) {
        pciRatingBadge.className = "tech-tag status-good";
      } else if (m.pci_score >= 60) {
        pciRatingBadge.className = "tech-tag status-fair";
      } else {
        pciRatingBadge.className = "tech-tag status-poor";
      }

      potholeCountEl.textContent = m.pothole_count;
      potholeAreaText.textContent = `Area: ${m.total_pothole_area_px.toLocaleString()} px² (${m.pothole_coverage_pct}%)`;

      crackDensityEl.textContent = `${m.crack_density_pct} %`;
      fatigueStatusText.textContent = m.is_fatigue_alligator ? "Fatigue: Alligator cracking detected" : "Fatigue: Sound surface";
      fatigueStatusText.className = m.is_fatigue_alligator ? "text-[11px] text-[#e5484d] mt-1" : "text-[11px] text-[#85858b] mt-1";

      actionRecommendationEl.textContent = m.action_recommendation;

      // Update Potholes List
      potholeListEl.innerHTML = "";
      if (m.potholes && m.potholes.length > 0) {
        m.potholes.forEach(p => {
          const div = document.createElement("div");
          div.className = "panel-raised p-2.5 flex items-center justify-between text-xs";
          div.innerHTML = `
            <div>
              <span class="font-medium text-[#ededed]">Pothole #${p.id}</span>
              <span class="text-[11px] text-[#85858b] block font-mono mt-0.5">${p.approx_size_cm} · ${p.severity}</span>
            </div>
            <span class="font-mono tabular-nums text-xs text-[#ededed]">${p.area_px.toLocaleString()} px²</span>
          `;
          potholeListEl.appendChild(div);
        });
      } else {
        potholeListEl.innerHTML = `<div class="p-4 text-center text-[#85858b] text-xs">No pothole cavities detected on this pavement surface</div>`;
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
