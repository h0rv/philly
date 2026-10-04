import logo from '../../../assets/philly.svg?raw';
export function GET() {
  const icon = logo.replace('viewBox="0 0 190 50"', 'viewBox="-10 0 50 60"')
    .replace('<title', '<style>svg{color:#222}@media(prefers-color-scheme:dark){svg{color:#eee}}</style><title');
  return new Response(icon, { headers: { 'content-type': 'image/svg+xml' } });
}
