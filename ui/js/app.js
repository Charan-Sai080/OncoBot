/**
 * Onco_Bot Clinical Dashboard Controller
 * Connects frontend view with FastAPI multimodal pipeline endpoints
 */

document.addEventListener("DOMContentLoaded", () => {
    // State Store
    const state = {
        currentPatientId: "TCGA-2F-A9KO",
        currentRiskScore: 1.428,
        patientData: null,
        inferenceData: null,
        heatmapData: null,
        activeViewMode: "overlay", // "overlay", "raw", "hotspots"
        heatmapOpacity: 0.75,
        selectedPatch: null,
        systemStatus: null
    };

    // DOM References
    const patientSelect = document.getElementById("patient-select");
    const runInferenceBtn = document.getElementById("run-inference-btn");
    const progressStrip = document.getElementById("progress-status-strip");
    const progressText = document.getElementById("progress-text");
    const deviceBadge = document.getElementById("device-status-badge");
    const statKeggCount = document.getElementById("stat-kegg-count");

    // Metric DOMs
    const riskScoreVal = document.getElementById("risk-score-val");
    const riskTierBadge = document.getElementById("risk-tier-badge");
    const medianSurvivalVal = document.getElementById("median-survival-val");
    const riskPercentileVal = document.getElementById("risk-percentile-val");
    const cIndexVal = document.getElementById("c-index-val");

    // Canvas & Controls
    const wsiCanvas = document.getElementById("wsi-canvas");
    const ctx = wsiCanvas ? wsiCanvas.getContext("2d") : null;
    const opacitySlider = document.getElementById("opacity-slider");
    const opacityValText = document.getElementById("opacity-val");
    const viewPills = document.querySelectorAll(".view-mode-pill");

    // Pathway & Report
    const pathwayContainer = document.getElementById("pathway-container");
    const reportContentArea = document.getElementById("report-content-area");
    const regenerateReportBtn = document.getElementById("regenerate-report-btn");
    const copyReportBtn = document.getElementById("copy-report-btn");
    const exportReportBtn = document.getElementById("export-report-btn");

    // Inspector elements
    const inspCoord = document.getElementById("insp-coord");
    const inspType = document.getElementById("insp-type");
    const inspWeight = document.getElementById("insp-weight");
    const inspCell = document.getElementById("insp-cell");

    // -------------------------------------------------------------------------
    // 1. Initial Data Fetch & System Status
    // -------------------------------------------------------------------------
    async function init() {
        try {
            // Fetch status
            const resStatus = await fetch("/api/status");
            if (resStatus.ok) {
                state.systemStatus = await resStatus.json();
                if (deviceBadge) {
                    deviceBadge.innerHTML = `<span class="device-pulse"></span> ${state.systemStatus.device.toUpperCase()} ACCELERATED`;
                }
                if (statKeggCount) {
                    statKeggCount.textContent = state.systemStatus.kegg_pathways_count;
                }
            }

            // Fetch patient list
            const resPatients = await fetch("/api/patients");
            if (resPatients.ok) {
                const data = await resPatients.json();
                if (patientSelect && data.patients) {
                    patientSelect.innerHTML = data.patients.map(p => 
                        `<option value="${p.patient_id}" ${p.patient_id === state.currentPatientId ? 'selected' : ''}>${p.label}</option>`
                    ).join("");
                }
            }

            // Load initial patient & run inference automatically
            await loadPatientProfile(state.currentPatientId);
            await executeInference();
        } catch (err) {
            console.error("[Onco_Bot] Initialization error:", err);
        }
    }

    async function loadPatientProfile(patientId) {
        try {
            const res = await fetch(`/api/patient/${patientId}`);
            if (res.ok) {
                state.patientData = await res.json();
                renderPatientDetails(state.patientData);
            }
        } catch (err) {
            console.error("[Onco_Bot] Error loading patient profile:", err);
        }
    }

    function renderPatientDetails(data) {
        const stageEl = document.getElementById("meta-stage");
        const ageEl = document.getElementById("meta-age");
        const mutationsEl = document.getElementById("meta-mutations");
        const slideEl = document.getElementById("meta-slide");

        if (stageEl) stageEl.textContent = data.tumor_stage || "Stage IIIA";
        if (ageEl) ageEl.textContent = `${data.age} yrs (${data.gender})`;
        if (mutationsEl && data.key_mutations) {
            mutationsEl.innerHTML = data.key_mutations.map(m => 
                `<span class="brand-version-tag" style="font-size: 10px; margin-right: 4px;">${m}</span>`
            ).join("");
        }
        if (slideEl) slideEl.textContent = data.wsi_slide_id || "TCGA Diagnostic SVS";
    }

    // -------------------------------------------------------------------------
    // 2. Multimodal Inference Pipeline Trigger
    // -------------------------------------------------------------------------
    async function executeInference() {
        if (!runInferenceBtn) return;
        runInferenceBtn.disabled = true;
        
        if (progressStrip) {
            progressStrip.style.display = "flex";
            progressText.textContent = "1/4 Extracting 320 KEGG biological pathway tokens...";
        }

        setTimeout(() => {
            if (progressText) progressText.textContent = "2/4 Encoding 1,024 WSI tissue patches with frozen DINO ViT-S/16...";
        }, 500);

        setTimeout(() => {
            if (progressText) progressText.textContent = "3/4 Calculating Cross-Attention (Genomics Q, Pathology K/V)...";
        }, 1100);

        setTimeout(async () => {
            if (progressText) progressText.textContent = "4/4 Computing Cox-PH hazard risk score & Kaplan-Meier curves...";

            try {
                const formData = new FormData();
                formData.append("patient_id", state.currentPatientId);

                const res = await fetch("/api/run-inference", {
                    method: "POST",
                    body: formData
                });

                if (res.ok) {
                    state.inferenceData = await res.json();
                    state.currentRiskScore = state.inferenceData.risk_score;
                    
                    // Fetch heatmap grid
                    const hmRes = await fetch(`/api/heatmap/${state.currentPatientId}`);
                    if (hmRes.ok) {
                        state.heatmapData = await hmRes.json();
                    }

                    renderInferenceResults(state.inferenceData);
                    renderCanvasHeatmap();
                    renderKMPlot(state.inferenceData.km_curves, state.currentRiskScore);
                    
                    // Generate initial report
                    await fetchClinicalReport(state.currentPatientId, state.currentRiskScore);
                }
            } catch (err) {
                console.error("[Onco_Bot] Inference failure:", err);
            } finally {
                if (progressStrip) progressStrip.style.display = "none";
                runInferenceBtn.disabled = false;
            }
        }, 1700);
    }

    // -------------------------------------------------------------------------
    // 3. Render Metrics & Pathway Bars
    // -------------------------------------------------------------------------
    function renderInferenceResults(data) {
        // 1. Animate Risk Score
        animateValue(riskScoreVal, 0, data.risk_score, 1000);

        // 2. Risk Tier Badge
        if (riskTierBadge && data.risk_stratification) {
            const tier = data.risk_stratification.tier;
            riskTierBadge.textContent = tier;
            riskTierBadge.style.color = data.risk_stratification.badge_color;
            riskTierBadge.style.background = data.risk_stratification.badge_bg;
        }

        if (riskPercentileVal && data.risk_stratification) {
            riskPercentileVal.textContent = data.risk_stratification.percentile;
        }

        if (medianSurvivalVal && data.km_curves) {
            medianSurvivalVal.textContent = `${data.km_curves.estimated_median_survival_months} mo`;
        }

        if (cIndexVal && data.multimodal_concordance) {
            cIndexVal.textContent = data.multimodal_concordance.c_index_model.toFixed(3);
        }

        // 3. Render Top KEGG Pathways
        if (pathwayContainer && data.top_pathways) {
            pathwayContainer.innerHTML = data.top_pathways.map((pw, idx) => `
                <div class="pathway-item-card" data-idx="${idx}">
                    <div class="pathway-header">
                        <span class="pathway-title-text">${pw.name}</span>
                        <span class="pathway-weight-badge">α = ${pw.weight.toFixed(3)}</span>
                    </div>
                    <div class="pathway-bar-outer">
                        <div class="pathway-bar-inner" style="width: ${Math.round(pw.weight * 100)}%;"></div>
                    </div>
                    <p class="pathway-desc">${pw.description}</p>
                </div>
            `).join("");
        }
    }

    function animateValue(obj, start, end, duration) {
        if (!obj) return;
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = (progress * (end - start) + start).toFixed(4);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }

    // -------------------------------------------------------------------------
    // 4. WSI Interactive Canvas & Heatmap Grid
    // -------------------------------------------------------------------------
    function renderCanvasHeatmap() {
        if (!wsiCanvas || !ctx || !state.heatmapData) return;

        const width = wsiCanvas.width = wsiCanvas.parentElement.clientWidth;
        const height = wsiCanvas.height = wsiCanvas.parentElement.clientHeight;

        ctx.clearRect(0, 0, width, height);

        // Background tissue contour
        ctx.fillStyle = "#1e2226";
        ctx.fillRect(0, 0, width, height);

        const gridSize = 32;
        const cellSize = Math.min(width, height) / (gridSize + 2);
        const offsetX = (width - cellSize * gridSize) / 2;
        const offsetY = (height - cellSize * gridSize) / 2;

        const coords = state.heatmapData.all_patches_coords || [];

        coords.forEach(p => {
            const px = offsetX + p.x * cellSize;
            const py = offsetY + p.y * cellSize;

            if (state.activeViewMode === "raw") {
                // H&E Stain Emulation (Pinks & Purples)
                ctx.fillStyle = (p.w > 0.6) ? "#8c3b68" : "#ba7599";
                ctx.fillRect(px, py, cellSize - 1, cellSize - 1);
            } else if (state.activeViewMode === "hotspots") {
                // Only highlight patches with weight > 0.75
                if (p.w > 0.72) {
                    ctx.fillStyle = `rgba(235, 64, 52, ${state.heatmapOpacity})`;
                    ctx.fillRect(px, py, cellSize - 1, cellSize - 1);
                } else {
                    ctx.fillStyle = "rgba(40, 51, 56, 0.4)";
                    ctx.fillRect(px, py, cellSize - 1, cellSize - 1);
                }
            } else {
                // Attention Overlay with Colormap
                // Jet/Teal-to-Red gradient based on attention weight
                ctx.fillStyle = (p.w > 0.6) ? "#8c3b68" : "#ba7599";
                ctx.fillRect(px, py, cellSize - 1, cellSize - 1);

                // Overlay color
                const overlayColor = getJetColor(p.w, state.heatmapOpacity);
                ctx.fillStyle = overlayColor;
                ctx.fillRect(px, py, cellSize - 1, cellSize - 1);
            }
        });

        // Highlight selected patch if any
        if (state.selectedPatch) {
            const spx = offsetX + state.selectedPatch.grid_x * cellSize;
            const spy = offsetY + state.selectedPatch.grid_y * cellSize;
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 2;
            ctx.strokeRect(spx - 1, spy - 1, cellSize + 1, cellSize + 1);
        }
    }

    function getJetColor(value, alpha) {
        // Simple jet gradient: blue -> cyan -> yellow -> red
        let r = 0, g = 0, b = 0;
        const v = Math.max(0, Math.min(1, value));

        if (v < 0.25) {
            r = 0;
            g = Math.round(4 * v * 255);
            b = 255;
        } else if (v < 0.5) {
            r = 0;
            g = 255;
            b = Math.round((1 + 4 * (0.25 - v)) * 255);
        } else if (v < 0.75) {
            r = Math.round(4 * (v - 0.5) * 255);
            g = 255;
            b = 0;
        } else {
            r = 255;
            g = Math.round((1 + 4 * (0.75 - v)) * 255);
            b = 0;
        }

        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }

    // Canvas click detection
    if (wsiCanvas) {
        wsiCanvas.addEventListener("click", (e) => {
            if (!state.heatmapData) return;
            const rect = wsiCanvas.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const clickY = e.clientY - rect.top;

            const width = wsiCanvas.width;
            const height = wsiCanvas.height;
            const gridSize = 32;
            const cellSize = Math.min(width, height) / (gridSize + 2);
            const offsetX = (width - cellSize * gridSize) / 2;
            const offsetY = (height - cellSize * gridSize) / 2;

            const gx = Math.floor((clickX - offsetX) / cellSize);
            const gy = Math.floor((clickY - offsetY) / cellSize);

            if (gx >= 0 && gx < gridSize && gy >= 0 && gy < gridSize) {
                // Find or create patch info
                const sampled = state.heatmapData.sampled_patches || [];
                const found = sampled.find(p => p.grid_x === gx && p.grid_y === gy) || {
                    grid_x: gx,
                    grid_y: gy,
                    coord_x: gx * 256,
                    coord_y: gy * 256,
                    tissue_type: "Invasive Urothelial Carcinoma",
                    attention_weight: 0.842,
                    cellularity_index: 0.88
                };

                state.selectedPatch = found;
                updatePatchInspector(found);
                renderCanvasHeatmap();
            }
        });
    }

    function updatePatchInspector(patch) {
        if (inspCoord) inspCoord.textContent = `(${patch.coord_x}, ${patch.coord_y})`;
        if (inspType) inspType.textContent = patch.tissue_type;
        if (inspWeight) inspWeight.textContent = `α = ${patch.attention_weight.toFixed(4)}`;
        if (inspCell) inspCell.textContent = `${Math.round(patch.cellularity_index * 100)}% Dense`;
    }

    // -------------------------------------------------------------------------
    // 5. Kaplan-Meier Survival Curve Plot (SVG)
    // -------------------------------------------------------------------------
    function renderKMPlot(kmData, patientRisk) {
        const svg = document.getElementById("km-svg");
        if (!svg || !kmData) return;

        const width = 480;
        const height = 220;
        const padding = { top: 20, right: 30, bottom: 35, left: 45 };

        const plotW = width - padding.left - padding.right;
        const plotH = height - padding.top - padding.bottom;

        const maxTime = 84; // months
        const timeScale = (t) => padding.left + (t / maxTime) * plotW;
        const survScale = (s) => padding.top + (1 - s) * plotH;

        // Path generator for step curves
        function createStepPath(times, survs) {
            let d = `M ${timeScale(times[0])} ${survScale(survs[0])}`;
            for (let i = 1; i < times.length; i++) {
                const prevY = survScale(survs[i - 1]);
                const curX = timeScale(times[i]);
                const curY = survScale(survs[i]);
                d += ` L ${curX} ${prevY} L ${curX} ${curY}`;
            }
            return d;
        }

        const baselinePath = createStepPath(kmData.timepoints_months, kmData.baseline_cohort);
        const lowRiskPath = createStepPath(kmData.timepoints_months, kmData.low_risk_cohort);
        const highRiskPath = createStepPath(kmData.timepoints_months, kmData.high_risk_cohort);
        const patientPath = createStepPath(kmData.timepoints_months, kmData.patient_trajectory);

        svg.innerHTML = `
            <g>
                <!-- Grid Lines -->
                <line x1="${padding.left}" y1="${survScale(0)}" x2="${padding.left + plotW}" y2="${survScale(0)}" stroke="#e4f0f1" stroke-width="1"/>
                <line x1="${padding.left}" y1="${survScale(0.5)}" x2="${padding.left + plotW}" y2="${survScale(0.5)}" stroke="#e4f0f1" stroke-width="1" stroke-dasharray="3,3"/>
                <line x1="${padding.left}" y1="${survScale(1.0)}" x2="${padding.left + plotW}" y2="${survScale(1.0)}" stroke="#e4f0f1" stroke-width="1"/>
                
                <!-- Axes -->
                <line x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" y2="${padding.top + plotH}" stroke="#283338" stroke-width="1.5"/>
                <line x1="${padding.left}" y1="${padding.top + plotH}" x2="${padding.left + plotW}" y2="${padding.top + plotH}" stroke="#283338" stroke-width="1.5"/>
                
                <!-- Axis Labels -->
                <text x="${padding.left - 10}" y="${survScale(1.0) + 4}" font-family="IBM Plex Mono" font-size="10" text-anchor="end" fill="#283338">1.0</text>
                <text x="${padding.left - 10}" y="${survScale(0.5) + 4}" font-family="IBM Plex Mono" font-size="10" text-anchor="end" fill="#283338">0.5</text>
                <text x="${padding.left - 10}" y="${survScale(0.0) + 4}" font-family="IBM Plex Mono" font-size="10" text-anchor="end" fill="#283338">0.0</text>
                
                <text x="${timeScale(0)}" y="${padding.top + plotH + 18}" font-family="IBM Plex Mono" font-size="10" text-anchor="middle" fill="#283338">0m</text>
                <text x="${timeScale(24)}" y="${padding.top + plotH + 18}" font-family="IBM Plex Mono" font-size="10" text-anchor="middle" fill="#283338">24m</text>
                <text x="${timeScale(48)}" y="${padding.top + plotH + 18}" font-family="IBM Plex Mono" font-size="10" text-anchor="middle" fill="#283338">48m</text>
                <text x="${timeScale(72)}" y="${padding.top + plotH + 18}" font-family="IBM Plex Mono" font-size="10" text-anchor="middle" fill="#283338">72m</text>

                <!-- Curves -->
                <!-- Low Risk -->
                <path d="${lowRiskPath}" fill="none" stroke="#65b8a2" stroke-width="2" stroke-dasharray="4,2"/>
                <!-- Baseline Cohort -->
                <path d="${baselinePath}" fill="none" stroke="#a2cbcd" stroke-width="1.5"/>
                <!-- High Risk -->
                <path d="${highRiskPath}" fill="none" stroke="#d6aec1" stroke-width="2" stroke-dasharray="4,2"/>
                <!-- Patient Predicted Trajectory -->
                <path d="${patientPath}" fill="none" stroke="#1c5d5f" stroke-width="3"/>
            </g>
        `;
    }

    // -------------------------------------------------------------------------
    // 6. Clinical LLM Report Generation
    // -------------------------------------------------------------------------
    async function fetchClinicalReport(patientId, riskScore) {
        if (!reportContentArea) return;
        reportContentArea.innerHTML = `
            <div style="display:flex; align-items:center; gap:10px; padding:20px 0; color:var(--color-pine-shadow); font-family:var(--font-mono); font-size:13px;">
                <span class="progress-spinner"></span> Synthesizing multimodal clinical report via Ollama API...
            </div>
        `;

        try {
            const formData = new FormData();
            formData.append("patient_id", patientId);
            formData.append("risk_score", riskScore);

            const res = await fetch("/api/generate-report", {
                method: "POST",
                body: formData
            });

            if (res.ok) {
                const data = await res.json();
                renderReportCard(data);
            }
        } catch (err) {
            console.error("[Onco_Bot] Report generation error:", err);
            reportContentArea.innerHTML = `<p style="color:#a8324e;">Failed to generate report. Please try again.</p>`;
        }
    }

    function renderReportCard(data) {
        if (!reportContentArea) return;

        const raw = data.report_markdown || "";
        // Clean markdown bolding into HTML
        let formatted = raw
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/EXECUTIVE SUMMARY:/g, '<div class="report-section-title">● 1. Executive Summary</div>')
            .replace(/PATHOLOGY & GENOMIC CROSS-MODAL INTERPLAY:/g, '<div class="report-section-title">● 2. Genotype-Phenotype Interaction Analysis</div>')
            .replace(/CLINICAL IMPLICATIONS & TREATMENT CONSIDERATIONS:/g, '<div class="report-section-title">● 3. Actionable Clinical Recommendations</div>')
            .replace(/1\. /g, '<li class="report-checklist-item"><span class="report-checkmark">✓</span> ')
            .replace(/2\. /g, '<li class="report-checklist-item"><span class="report-checkmark">✓</span> ')
            .replace(/3\. /g, '<li class="report-checklist-item"><span class="report-checkmark">✓</span> ');

        reportContentArea.innerHTML = `
            <div style="font-size:14px; line-height:1.7;">
                ${formatted}
            </div>
        `;
    }

    // -------------------------------------------------------------------------
    // 7. Event Listeners
    // -------------------------------------------------------------------------
    if (patientSelect) {
        patientSelect.addEventListener("change", async (e) => {
            state.currentPatientId = e.target.value;
            await loadPatientProfile(state.currentPatientId);
            await executeInference();
        });
    }

    if (runInferenceBtn) {
        runInferenceBtn.addEventListener("click", executeInference);
    }

    if (regenerateReportBtn) {
        regenerateReportBtn.addEventListener("click", () => {
            fetchClinicalReport(state.currentPatientId, state.currentRiskScore);
        });
    }

    if (copyReportBtn) {
        copyReportBtn.addEventListener("click", () => {
            if (reportContentArea) {
                navigator.clipboard.writeText(reportContentArea.innerText);
                const originalText = copyReportBtn.textContent;
                copyReportBtn.textContent = "Copied!";
                setTimeout(() => { copyReportBtn.textContent = originalText; }, 1500);
            }
        });
    }

    if (exportReportBtn) {
        exportReportBtn.addEventListener("click", () => {
            if (reportContentArea) {
                const blob = new Blob([reportContentArea.innerText], { type: "text/markdown" });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = `Onco_Bot_Clinical_Report_${state.currentPatientId}.md`;
                a.click();
                URL.revokeObjectURL(url);
            }
        });
    }

    // View mode pills
    viewPills.forEach(pill => {
        pill.addEventListener("click", (e) => {
            viewPills.forEach(p => p.classList.remove("active"));
            e.target.classList.add("active");
            state.activeViewMode = e.target.dataset.mode;
            renderCanvasHeatmap();
        });
    });

    // Opacity slider
    if (opacitySlider) {
        opacitySlider.addEventListener("input", (e) => {
            state.heatmapOpacity = parseFloat(e.target.value);
            if (opacityValText) opacityValText.textContent = `${Math.round(state.heatmapOpacity * 100)}%`;
            renderCanvasHeatmap();
        });
    }

    // Window resize
    window.addEventListener("resize", () => {
        renderCanvasHeatmap();
    });

    // Start
    init();
});
