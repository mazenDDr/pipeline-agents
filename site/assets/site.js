const root = document.documentElement;
root.dataset.theme = matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
document.querySelector('#theme').addEventListener('click', () => {
  root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
});
document.querySelectorAll('figure').forEach(figure => {
  let image = figure.querySelector('img');
  if (!image) return;
  const button = document.createElement('button');
  button.textContent = 'Replay figure';
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const syncMotion = () => {
    button.disabled = reducedMotion.matches;
    button.textContent = reducedMotion.matches ? 'Reduced motion enabled' : 'Replay figure';
  };
  reducedMotion.addEventListener('change', syncMotion);
  syncMotion();
  button.addEventListener('click', () => {
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const next = image.cloneNode();
    image.replaceWith(next);
    // Reloading this local asset restarts the self-contained SVG timeline.
    next.src = image.src.split('?')[0] + '?replay=' + Date.now();
    image = next;
  });
  figure.append(button);
  const observer = new IntersectionObserver(entries => {
    if (!entries.some(entry => entry.isIntersecting)) return;
    if (!reducedMotion.matches) button.click();
    observer.disconnect();
  }, {threshold: 0.25});
  observer.observe(figure);
});
