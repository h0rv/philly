// Service Worker — caches ArcGIS tiles indefinitely.
// Historical imagery never changes, so we can cache forever.

const CACHE = "philly-tiles-v1";
const TILE_HOST = "tiles.arcgis.com";

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(clients.claim()));

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (url.hostname !== TILE_HOST) return;

  e.respondWith(
    caches.open(CACHE).then((cache) =>
      cache.match(e.request).then((cached) => {
        if (cached) return cached;
        return fetch(e.request).then((res) => {
          if (res.ok) cache.put(e.request, res.clone());
          return res;
        });
      }),
    ),
  );
});
