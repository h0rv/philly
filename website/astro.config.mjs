import { defineConfig } from 'astro/config';

const repoName = process.env.GITHUB_REPOSITORY?.split('/')[1] ?? 'philly';
const owner = process.env.GITHUB_REPOSITORY_OWNER ?? process.env.GITHUB_REPOSITORY?.split('/')[0];
const isGitHubPagesBuild = process.env.GITHUB_ACTIONS === 'true';
const siteUrl = process.env.SITE_URL;
const hasCustomSiteUrl = typeof siteUrl === 'string' && siteUrl.length > 0;

export default defineConfig({
  output: 'static',
  trailingSlash: 'always',
  base: isGitHubPagesBuild && !hasCustomSiteUrl ? `/${repoName}/` : '/',
  site: hasCustomSiteUrl ? siteUrl : owner ? `https://${owner}.github.io` : 'http://localhost:4321',
});
