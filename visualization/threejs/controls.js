export function setupControls(data, cityRenderer, updateStats) {
  const slider = document.getElementById('step-slider');
  const playBtn = document.getElementById('play');
  const pauseBtn = document.getElementById('pause');
  const stepLabel = document.getElementById('step-label');

  let currentStep = 0;
  let playing = false;
  let interval = null;

  function goToStep(idx) {
    currentStep = Math.max(0, Math.min(idx, data.length - 1));
    slider.value = String(currentStep);
    stepLabel.textContent = `Step: ${currentStep}`;
    cityRenderer.render(data[currentStep]);
    updateStats(data[currentStep]);
  }

  playBtn.addEventListener('click', () => {
    if (playing) return;
    playing = true;
    interval = window.setInterval(() => {
      if (currentStep >= data.length - 1) {
        window.clearInterval(interval);
        playing = false;
        return;
      }
      goToStep(currentStep + 1);
    }, 200);
  });

  pauseBtn.addEventListener('click', () => {
    playing = false;
    window.clearInterval(interval);
  });

  slider.addEventListener('input', () => {
    goToStep(parseInt(slider.value, 10));
  });
}
