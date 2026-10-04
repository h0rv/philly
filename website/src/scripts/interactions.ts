document.querySelectorAll<HTMLButtonElement>('[data-copy]').forEach(button => {
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
