document.querySelectorAll<HTMLElement>('[data-tabs]').forEach(group => {
  const tabs = Array.from(group.querySelectorAll<HTMLButtonElement>('[role=tab]'));
  const select = (tab: HTMLButtonElement) => {
    tabs.forEach(item => {
      const selected = item === tab;
      item.setAttribute('aria-selected', String(selected)); item.tabIndex = selected ? 0 : -1;
      const panel = document.getElementById(item.getAttribute('aria-controls')!);
      if (panel) panel.hidden = !selected;
    });
  };
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => select(tab));
    tab.addEventListener('keydown', event => {
      const next = event.key === 'ArrowRight' ? (index + 1) % tabs.length : event.key === 'ArrowLeft' ? (index + tabs.length - 1) % tabs.length : event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : null;
      if (next !== null) { event.preventDefault(); select(tabs[next]); tabs[next].focus(); }
    });
  });
});
document.querySelectorAll<HTMLButtonElement>('.copy').forEach(button => {
  button.addEventListener('click', async () => {
    const code = button.parentElement?.querySelector('code');
    const status = document.querySelector('[data-copy-status]');
    try {
      await navigator.clipboard.writeText(code?.textContent ?? ''); button.textContent = 'Copied';
      if (status) status.textContent = 'Command copied to clipboard.';
    } catch {
      button.textContent = 'Select text';
      if (status) status.textContent = 'Clipboard unavailable. Select and copy the code below.';
      if (code) { const range = document.createRange(); range.selectNodeContents(code); const selection = window.getSelection(); selection?.removeAllRanges(); selection?.addRange(range); }
    }
    setTimeout(() => { button.textContent = 'Copy'; }, 2500);
  });
});
