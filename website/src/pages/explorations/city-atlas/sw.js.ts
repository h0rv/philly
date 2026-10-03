import workerSource from '../../../../public/sw.js?raw';
// Serve the existing tile cache worker in the moved Atlas page's own scope.
export function GET() {
  return new Response(workerSource, { headers: { 'content-type': 'text/javascript; charset=utf-8' } });
}
