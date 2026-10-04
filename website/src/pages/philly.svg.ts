import logo from '../../../assets/philly.svg?raw';
export function GET() {
  return new Response(logo, { headers: { 'content-type': 'image/svg+xml' } });
}
