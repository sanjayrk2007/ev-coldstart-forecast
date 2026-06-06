import React, { useEffect, useRef } from 'react';
import L from 'leaflet';

export default function GeospatialMap({ lat, lng, onMapClick, theme = 'dark' }) {
  const mapContainerRef = useRef(null);
  const mapRef = useRef(null);
  const markerRef = useRef(null);

  // Initialize Map
  useEffect(() => {
    if (!mapContainerRef.current) return;

    // Create map instance centered at lat/lng
    const map = L.map(mapContainerRef.current, {
      center: [lat, lng],
      zoom: 13,
      zoomControl: false,
    });
    mapRef.current = map;

    // Zoom controls on bottom right
    L.control.zoom({ position: 'bottomright' }).addTo(map);

    // Map click handler to select new coordinates
    map.on('click', (e) => {
      if (onMapClick) {
        onMapClick(e.latlng.lat, e.latlng.lng);
      }
    });

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Update Tile Layer based on theme (Dark Matter vs Voyager)
  useEffect(() => {
    if (!mapRef.current) return;

    // Remove old tile layers if any
    mapRef.current.eachLayer((layer) => {
      if (layer instanceof L.TileLayer) {
        mapRef.current.removeLayer(layer);
      }
    });

    const tileUrl = theme === 'dark'
      ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
      : 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png';

    const attribution = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>';

    L.tileLayer(tileUrl, { attribution, maxZoom: 20 }).addTo(mapRef.current);
  }, [theme]);

  // Update Marker Position and Pan Map
  useEffect(() => {
    if (!mapRef.current) return;

    const position = [lat, lng];

    if (markerRef.current) {
      markerRef.current.setLatLng(position);
    } else {
      // Pulse animation marker in Electric Teal
      const pulseIcon = L.divIcon({
        className: 'custom-pulse-icon',
        html: `<div style="
          width: 14px;
          height: 14px;
          background-color: var(--color-primary, #41e0a6);
          border: 2px solid #ffffff;
          border-radius: 50%;
          box-shadow: 0 0 10px var(--color-primary, #41e0a6), 0 0 20px var(--color-primary, #41e0a6);
          animation: mapPulse 1.5s infinite;
        "></div>`,
        iconSize: [14, 14],
        iconAnchor: [7, 7],
      });

      markerRef.current = L.marker(position, { icon: pulseIcon }).addTo(mapRef.current);
    }

    mapRef.current.panTo(position);

  }, [lat, lng]);

  return (
    <div style={{ position: 'relative' }}>
      <div ref={mapContainerRef} className="map-container" />
      <style>{`
        .custom-pulse-icon {
          background: transparent;
          border: none;
        }
        @keyframes mapPulse {
          0% {
            transform: scale(0.9);
            box-shadow: 0 0 0 0 rgba(65, 224, 166, 0.7);
          }
          70% {
            transform: scale(1.2);
            box-shadow: 0 0 0 8px rgba(65, 224, 166, 0);
          }
          100% {
            transform: scale(0.9);
            box-shadow: 0 0 0 0 rgba(65, 224, 166, 0);
          }
        }
      `}</style>
    </div>
  );
}
