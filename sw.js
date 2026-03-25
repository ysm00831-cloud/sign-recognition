const CACHE_NAME = 'suhwa-v1';
const ASSETS = [
  '/sign-recognition/',
  '/sign-recognition/index.html',
  '/sign-recognition/jamo.html',
  '/sign-recognition/sentence.html',
  '/sign-recognition/manifest.json',
  '/sign-recognition/icons/icon-192.png',
  '/sign-recognition/icons/icon-512.png',
];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', e => {
  e.respondWith(
    caches.match(e.request).then(cached => cached || fetch(e.request))
  );
});
