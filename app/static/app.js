(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  /* ---------- header ---------- */
  const header = $("#siteHeader");
  const onScroll = () => header && header.classList.toggle("scrolled", window.scrollY > 12);
  onScroll();
  window.addEventListener("scroll", onScroll, { passive: true });

  const menuBtn = $("#menuBtn"), mobileNav = $("#mobileNav");
  if (menuBtn && mobileNav) {
    menuBtn.addEventListener("click", () => {
      const open = mobileNav.classList.toggle("open");
      menuBtn.setAttribute("aria-expanded", String(open));
    });
    $$("a", mobileNav).forEach((a) => a.addEventListener("click", () => mobileNav.classList.remove("open")));
  }

  /* ---------- reveal on scroll ---------- */
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
    }, { threshold: 0.15 });
    $$(".reveal").forEach((el) => io.observe(el));
  } else {
    $$(".reveal").forEach((el) => el.classList.add("in"));
  }

  /* ---------- copy button ---------- */
  $$("[data-copy]").forEach((btn) => btn.addEventListener("click", async () => {
    const target = $(btn.dataset.copy);
    const label = $(".copy-label", btn);
    if (!target) return;
    try {
      await navigator.clipboard.writeText(target.innerText);
      const old = label.textContent;
      label.textContent = "Copied";
      setTimeout(() => (label.textContent = old), 1600);
    } catch (_) {}
  }));

  /* ---------- detector (home page only) ---------- */
  const zone = $("#zone");
  if (!zone) return;

  const dropView = $("#dropView"), resultView = $("#resultView"), fileInput = $("#fileInput");
  const errBox = $("#errBox"), errTitle = $("#errTitle"), errMsg = $("#errMsg");
  const imgEl = $("#previewImg"), scan = $("#scan"), fileName = $("#fileName"), fileDims = $("#fileDims");
  const analyzing = $("#analyzing"), card = $("#resultCard"), zoneTitle = $("#zoneTitle");
  const ring = $("#ringArc"), ringNum = $("#ringNum");
  //const ENDPOINT = $("#resultView") ? "{{ENDPOINT}}" : "";
  let depth = 0, objectUrl = null, controller = null;

  const restart = (el) => { el.classList.remove("view-in"); void el.offsetWidth; el.classList.add("view-in"); };

  function showView(state) {
    const isDrop = state === "drop";
    dropView.hidden = !isDrop;
    resultView.hidden = isDrop;
    if (isDrop) restart(dropView);
    analyzing.hidden = state !== "analyzing";
    scan.hidden = state !== "analyzing";
    card.hidden = state !== "result";
  }

  function showError(title, message) {
    errTitle.textContent = title;
    errMsg.textContent = " " + message;
    errBox.hidden = false;
  }

  /* drag and drop */
  const setDrag = (on) => {
    dropView.classList.toggle("dragging", on);
    zoneTitle.textContent = on ? "Release to analyze" : "Drop an image to analyze";
  };
  dropView.addEventListener("dragenter", (e) => { e.preventDefault(); depth++; setDrag(true); });
  dropView.addEventListener("dragover", (e) => e.preventDefault());
  dropView.addEventListener("dragleave", () => { depth = Math.max(0, depth - 1); if (depth === 0) setDrag(false); });
  dropView.addEventListener("drop", (e) => {
    e.preventDefault(); depth = 0; setDrag(false);
    handleFile(e.dataTransfer.files[0]);
  });
  // stop the browser opening a file that is dropped outside the zone
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  zone.addEventListener("click", () => fileInput.click());
  zone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener("change", () => { handleFile(fileInput.files[0]); fileInput.value = ""; });
  $("#againBtn").addEventListener("click", () => {
    if (controller) controller.abort();
    showView("drop");
  });

  function handleFile(file) {
    if (!file) return;
    errBox.hidden = true;
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);
    fileDims.textContent = "";
    imgEl.onload = () => (fileDims.textContent = `${imgEl.naturalWidth}×${imgEl.naturalHeight}`);
    imgEl.src = objectUrl;
    fileName.textContent = file.name;
    showView("analyzing");

    // The SERVER validates the file (type, size, decoding) and runs the model.
    const body = new FormData();
    body.append("image", file);
    if (controller) controller.abort();
    controller = new AbortController();
    const started = performance.now();

    fetch(zone.dataset.endpoint, {
      method: "POST", body, headers: { "X-Requested-With": "fetch" }, signal: controller.signal,
    })
      .then(async (r) => {
        let data = {};
        try { data = await r.json(); } catch (_) {}
        if (!r.ok) { const err = new Error(data.error || "Something went wrong. Please try again."); err.title = data.title; throw err; }
        return data;
      })
      .then(async (data) => {
        await sleep(Math.max(0, 1800 - (performance.now() - started))); // let the scan animation play
        render(data);
      })
      .catch((err) => {
        if (err.name === "AbortError") return;
        showView("drop");
        if (err instanceof TypeError) showError("Connection problem", "Could not reach the server. Please try again.");
        else showError(err.title || "We couldn't use that file", err.message);
      });
  }

  const fmtBytes = (b) => (b > 1048576 ? (b / 1048576).toFixed(2) + " MB" : Math.max(1, Math.round(b / 1024)) + " KB");

  function render(d) {
    const ai = d.label === "AI-generated";
    card.dataset.tone = ai ? "ai" : "real";
    $("#verdict").textContent = d.label;
    $("#verdictNote").textContent = ai
      ? "This image shows patterns typical of generative models."
      : "This image looks like a genuine, camera-captured photograph.";
    $("#verdictIcon").replaceChildren($(ai ? "#tpl-ai" : "#tpl-real").content.cloneNode(true));
    $("#dRes").textContent = `${d.width}×${d.height}`;
    $("#dSize").textContent = fmtBytes(d.file_size);
    $("#dModel").textContent = `${d.model} v${d.model_version}`;
    $("#dLat").textContent = `${Math.round(d.latency_ms)} ms`;
    const saved = $("#savedNote");
    saved.textContent = d.saved ? "Saved to history" : "Not saved to history";
    saved.classList.toggle("warn-chip", !d.saved);
    showView("result");
    animateRing(d.confidence * 100);
  }

  function animateRing(pct) {
    const C = 2 * Math.PI * ring.r.baseVal.value;
    ring.style.strokeDasharray = C;
    ring.style.transition = "none";
    ring.style.strokeDashoffset = C;
    void ring.getBoundingClientRect();
    ring.style.transition = "stroke-dashoffset 1.5s cubic-bezier(.22,1,.36,1)";
    ring.style.strokeDashoffset = C * (1 - pct / 100);
    const t0 = performance.now(), dur = 1500;
    const tick = (now) => {
      const p = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - p, 3);
      ringNum.textContent = (pct * e).toFixed(1);
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
})();