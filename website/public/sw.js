const CACHE = "philly-tiles-v1";
const TILE_HOST = "tiles.arcgis.com";

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(clients.claim()));

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (url.hostname !== TILE_HOST) {
    return;
  }

  event.respondWith(
    caches.open(CACHE).then((cache) =>
      cache.match(event.request).then((cached) => {
        if (cached) {
          return cached;
        }

        return fetch(event.request).then((response) => {
          if (response.ok) {
            cache.put(event.request, response.clone());
          }

          return response;
        });
      }),
    ),
  );
});
