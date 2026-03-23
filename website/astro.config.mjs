import { defineConfig } from 'astro/config';

const repoName = process.env.GITHUB_REPOSITORY?.split('/')[1] ?? 'philly';
const owner = process.env.GITHUB_REPOSITORY_OWNER ?? process.env.GITHUB_REPOSITORY?.split('/')[0];
const isGitHubPagesBuild = process.env.GITHUB_ACTIONS === 'true';

export default defineConfig({
  output: 'static',
  trailingSlash: 'always',
  base: isGitHubPagesBuild ? `/${repoName}/` : '/',
  site: owner ? `https://${owner}.github.io` : 'http://localhost:4321',
});
