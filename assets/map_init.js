// This file must export an arrow function string compatible with gradio's js hook.
() => {
    const root = (document.querySelector('gradio-app')?.shadowRoot) ?? document;

    const waitFor = (selector, { parent = root, timeout = 15000 } = {}) =>
        new Promise((resolve, reject) => {
            const check = () => {
                const el = parent.querySelector(selector);
                if (el) { resolve(el); return true; }
                return false;
            };
            if (check()) return;
            const obs = new MutationObserver(() => { if (check()) obs.disconnect(); });
            obs.observe(parent, { childList: true, subtree: true });
            setTimeout(() => { obs.disconnect(); reject(new Error('Timeout: ' + selector)); }, timeout);
        });

    const ensureLeaflet = () => new Promise((resolve, reject) => {
        if (window.L) return resolve();
        if (!document.getElementById('leaflet-css')) {
            const css = document.createElement('link');
            css.id = 'leaflet-css';
            css.rel = 'stylesheet';
            css.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
            document.head.appendChild(css);
        }
        const s = document.createElement('script');
        s.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
        s.onload = resolve;
        s.onerror = () => reject(new Error('Failed to load Leaflet'));
        document.head.appendChild(s);
    });

    (async () => {
        await ensureLeaflet();

        const mapEl = await waitFor('#map');
        const coordWrapper = await waitFor('#coord_input'); // wrapper div

        const coordField = await waitFor('textarea, input[type="text"]', { parent: coordWrapper });
        const map = L.map(mapEl).setView([49.258, -123.17], 8);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 18,
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);

        map.whenReady(() => map.invalidateSize());
        map.on('resize', () => map.invalidateSize());

        let marker;
        map.on('click', (e) => {
            if (marker) map.removeLayer(marker);
            marker = L.marker(e.latlng).addTo(map)
                .bindPopup(`Lat: ${e.latlng.lat.toFixed(4)}, Lng: ${e.latlng.lng.toFixed(4)}`)
                .openPopup();

            coordField.value = `${+e.latlng.lat},${+e.latlng.lng}`;
            coordField.dispatchEvent(new Event('input', { bubbles: true }));
        });
        const STUDY_BOUNDS = L.latLngBounds([[40.0, -160.0], [70.0, -110.0]]);
        L.rectangle(STUDY_BOUNDS, {
            color: '#222',
            weight: 1,
            dashArray: '4 4',
            fill: false
        }).addTo(map);
        map.setMaxBounds(STUDY_BOUNDS);
    })().catch(err => console.error('Map init failed:', err));

    return [];
}