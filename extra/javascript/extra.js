(function () {
  // Prevent installing the hook twice.
  if (window.__marimoBookMathJaxPrecomputeFix) {
    return;
  }
  window.__marimoBookMathJaxPrecomputeFix = true;

  let pendingFrame = null;
  let typesetChain = Promise.resolve();

  function handlePrecomputeChange(event) {
    const target = event.target;

    if (!(target instanceof Element)) {
      return;
    }

    // Only react to marimo-book's precomputed controls.
    if (!target.closest(".marimo-book-precompute-input")) {
      return;
    }

    const cells = Array.from(
      document.querySelectorAll(".marimo-book-precompute-cell")
    );

    // This event listener runs in the capture phase, i.e. before
    // marimo-book replaces the cell's innerHTML.
    if (window.MathJax?.typesetClear) {
      MathJax.typesetClear(cells);
    }

    // marimo-book performs the HTML replacement synchronously during
    // the same input event. Wait until the next animation frame and
    // typeset the newly inserted LaTeX.
    if (pendingFrame !== null) {
      cancelAnimationFrame(pendingFrame);
    }

    pendingFrame = requestAnimationFrame(() => {
      pendingFrame = null;

      if (!window.MathJax?.typesetPromise) {
        return;
      }

      const updatedCells = Array.from(
        document.querySelectorAll(".marimo-book-precompute-cell")
      );

      // Serialize MathJax calls in case the slider is moved quickly.
      typesetChain = typesetChain
        .then(() => MathJax.typesetPromise(updatedCells))
        .catch((error) => {
          console.error(
            "MathJax re-typesetting after marimo-book update failed:",
            error
          );
        });
    });
  }

  // Capture phase is important: this lets us clear MathJax before
  // marimo-book replaces the old mathematical content.
  document.addEventListener("input", handlePrecomputeChange, true);
  document.addEventListener("change", handlePrecomputeChange, true);
})();