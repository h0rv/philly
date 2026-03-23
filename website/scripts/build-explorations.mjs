import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { explorations } from '../config/explorations.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const websiteRoot = path.resolve(__dirname, '..');
const repoRoot = path.resolve(websiteRoot, '..');
const sourceRoot = path.join(repoRoot, 'explorations');
const publicRoot = path.join(websiteRoot, 'public');
const targetRoot = path.join(publicRoot, 'explorations');
const manifestPath = path.join(websiteRoot, 'src', 'data', 'explorations.generated.json');

async function exists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function ensureDir(filePath) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
}

async function copyEntry(sourceDir, targetDir, relativePath) {
  const sourcePath = path.join(sourceDir, relativePath);
  const targetPath = path.join(targetDir, relativePath);
  const stat = await fs.stat(sourcePath);

  if (stat.isDirectory()) {
    await fs.mkdir(targetPath, { recursive: true });
    const entries = await fs.readdir(sourcePath, { withFileTypes: true });
    for (const entry of entries) {
      await copyEntry(sourceDir, targetDir, path.join(relativePath, entry.name));
    }
    return;
  }

  await ensureDir(targetPath);
  await fs.copyFile(sourcePath, targetPath);
}

async function main() {
  await fs.mkdir(publicRoot, { recursive: true });
  await fs.rm(targetRoot, { recursive: true, force: true });
  await fs.mkdir(targetRoot, { recursive: true });

  const logoSource = path.join(repoRoot, 'assets', 'logo.png');
  const logoTarget = path.join(publicRoot, 'logo.png');
  if (await exists(logoSource)) {
    await fs.copyFile(logoSource, logoTarget);
  }

  const manifest = [];

  for (const exploration of explorations) {
    const sourceDir = path.join(sourceRoot, exploration.slug);
    const targetDir = path.join(targetRoot, exploration.slug);
    const requiredFiles = exploration.requiredFiles ?? ['index.html'];
    const missingFiles = [];

    for (const relativePath of requiredFiles) {
      if (!(await exists(path.join(sourceDir, relativePath)))) {
        missingFiles.push(relativePath);
      }
    }

    const ready = missingFiles.length === 0;

    if (ready) {
      for (const relativePath of exploration.publishFiles ?? requiredFiles) {
        await copyEntry(sourceDir, targetDir, relativePath);
      }
    }

    manifest.push({
      ...exploration,
      ready,
      missingFiles,
      href: `/explorations/${exploration.slug}/`,
    });
  }

  await ensureDir(manifestPath);
  await fs.writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);

  const readyCount = manifest.filter((item) => item.ready).length;
  const pendingCount = manifest.length - readyCount;
  console.log(`Synced ${readyCount} live explorations${pendingCount ? `, ${pendingCount} pending build` : ''}.`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
