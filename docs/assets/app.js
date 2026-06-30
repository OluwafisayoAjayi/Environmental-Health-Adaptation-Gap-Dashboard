const DATA_URL = 'data/dashboard_county_latest.csv';
const DEMO_URL = 'data/demo_dashboard_county.csv';
const META_URL = 'data/dashboard_metadata.json';

let allRows = [];
let filteredRows = [];
let markers = [];
let map;

function toNum(x){
  if (x === null || x === undefined || x === '') return null;
  const n = Number(x);
  return Number.isFinite(n) ? n : null;
}

function fmt(x, digits=1){
  const n = toNum(x);
  if (n === null) return '—';
  return Number(n).toLocaleString('en-US', { maximumFractionDigits: digits });
}

function countyLabel(r){
  const name = r.county_name || r.acs_name || r.county_name_shape || r.county_fips;
  const st = r.state || '';
  return st && !String(name).includes(',') ? `${name}, ${st}` : name;
}

function metricLabel(metric){
  const labels = {
    ehagi: 'EHAGI',
    climate_hazard_burden: 'Climate/natural hazard burden',
    air_pollution_exposure: 'Air pollution exposure',
    health_vulnerability: 'Health vulnerability',
    adaptive_capacity_deficit: 'Adaptive capacity deficit',
    risk_score: 'FEMA NRI risk score',
    max_aqi: 'Max AQI',
    pm25_mean: 'PM2.5 mean',
    asthma_prev: 'Asthma prevalence',
    poverty_rate: 'Poverty rate'
  };
  return labels[metric] || metric;
}

function colorScale(v, metric){
  const val = toNum(v);
  if (val === null) return '#cbd5e1';
  if (val >= 90) return '#7f1d1d';
  if (val >= 75) return '#b91c1c';
  if (val >= 50) return '#f97316';
  if (val >= 25) return '#eab308';
  return '#22c55e';
}

function initMap(){
  map = L.map('map', { scrollWheelZoom: false }).setView([39.5, -98.35], 4);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 18,
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(map);
  const legend = L.control({position:'bottomright'});
  legend.onAdd = function(){
    const div = L.DomUtil.create('div','legend');
    div.innerHTML = '<strong>Gap score</strong><br><i style="background:#7f1d1d"></i>Very high<br><i style="background:#b91c1c"></i>High<br><i style="background:#f97316"></i>Medium<br><i style="background:#22c55e"></i>Low';
    return div;
  };
  legend.addTo(map);
}

async function loadCsvWithFallback(){
  try {
    const resp = await fetch(DATA_URL, { cache: 'no-store' });
    if (!resp.ok) throw new Error('latest data not found');
    const text = await resp.text();
    return { text, demo: false };
  } catch (err) {
    const resp = await fetch(DEMO_URL, { cache: 'no-store' });
    const text = await resp.text();
    return { text, demo: true };
  }
}

async function loadData(){
  initMap();
  const loaded = await loadCsvWithFallback();
  const parsed = Papa.parse(loaded.text, { header: true, dynamicTyping: false, skipEmptyLines: true });
  allRows = parsed.data.map((r, idx) => ({...r, _idx: idx}));

  let metaText = loaded.demo ? 'Demo preview data loaded. Run the GitHub Action to pull live public datasets.' : 'Live generated dashboard data loaded.';
  try {
    const meta = await fetch(META_URL, { cache: 'no-store' }).then(r => r.ok ? r.json() : null);
    if (meta && meta.built_at_utc && meta.built_at_utc !== 'demo-preview') metaText = `Last updated: ${new Date(meta.built_at_utc).toLocaleString()} | Rows: ${meta.rows?.toLocaleString?.() || meta.rows}`;
  } catch(e) {}
  document.getElementById('lastUpdated').textContent = metaText;

  populateFilters();
  applyFilters();
}

function populateFilters(){
  const stateSelect = document.getElementById('stateFilter');
  const states = [...new Set(allRows.map(r => r.state).filter(Boolean))].sort();
  for (const st of states){
    const opt = document.createElement('option'); opt.value = st; opt.textContent = st; stateSelect.appendChild(opt);
  }
  const prioritySelect = document.getElementById('priorityFilter');
  const groups = [...new Set(allRows.map(r => r.priority_group).filter(Boolean))].sort();
  for (const g of groups){
    const opt = document.createElement('option'); opt.value = g; opt.textContent = g; prioritySelect.appendChild(opt);
  }

  ['stateFilter','metricFilter','priorityFilter','countySearch'].forEach(id => {
    document.getElementById(id).addEventListener(id === 'countySearch' ? 'input' : 'change', applyFilters);
  });
  document.getElementById('resetBtn').addEventListener('click', () => {
    document.getElementById('stateFilter').value = 'ALL';
    document.getElementById('metricFilter').value = 'ehagi';
    document.getElementById('priorityFilter').value = 'ALL';
    document.getElementById('countySearch').value = '';
    applyFilters();
  });
}

function applyFilters(){
  const st = document.getElementById('stateFilter').value;
  const group = document.getElementById('priorityFilter').value;
  const q = document.getElementById('countySearch').value.trim().toLowerCase();

  filteredRows = allRows.filter(r => {
    const stOK = st === 'ALL' || r.state === st;
    const gOK = group === 'ALL' || r.priority_group === group;
    const text = `${r.county_name || ''} ${r.acs_name || ''} ${r.state || ''} ${r.county_fips || ''}`.toLowerCase();
    const qOK = !q || text.includes(q);
    return stOK && gOK && qOK;
  });
  updateKPIs();
  updateMap();
  updateTable();
  updateScatter();
}

function average(rows, field){
  const vals = rows.map(r => toNum(r[field])).filter(v => v !== null);
  if (!vals.length) return null;
  return vals.reduce((a,b)=>a+b,0)/vals.length;
}

function updateKPIs(){
  const rows = filteredRows;
  const high = rows.filter(r => (r.priority_group || '').startsWith('Highest adaptation gap')).length;
  document.getElementById('kpiCounties').textContent = rows.length.toLocaleString();
  document.getElementById('kpiAvgIndex').textContent = fmt(average(rows, 'ehagi'));
  document.getElementById('kpiHighPriority').textContent = high.toLocaleString();
  document.getElementById('kpiPollution').textContent = fmt(average(rows, 'air_pollution_exposure'));
}

function updateMap(){
  const metric = document.getElementById('metricFilter').value;
  markers.forEach(m => map.removeLayer(m));
  markers = [];
  const pts = filteredRows.filter(r => toNum(r.lat)!==null && toNum(r.lon)!==null);
  for (const r of pts){
    const val = toNum(r[metric]);
    const radius = Math.max(4, Math.min(15, 4 + (toNum(r.ehagi)||0)/10));
    const marker = L.circleMarker([toNum(r.lat), toNum(r.lon)], {
      radius,
      color: '#ffffff',
      weight: 1,
      fillColor: colorScale(val, metric),
      fillOpacity: 0.82
    }).addTo(map);
    marker.bindPopup(`<strong>${countyLabel(r)}</strong><br>${metricLabel(metric)}: ${fmt(val)}<br>EHAGI: ${fmt(r.ehagi)}<br>Priority: ${r.priority_group || '—'}`);
    marker.on('click', () => showProfile(r));
    markers.push(marker);
  }
  if (pts.length && document.getElementById('stateFilter').value !== 'ALL') {
    const bounds = L.latLngBounds(pts.map(r => [toNum(r.lat), toNum(r.lon)]));
    map.fitBounds(bounds.pad(0.25));
  }
}

function updateTable(){
  const tbody = document.querySelector('#topTable tbody');
  tbody.innerHTML = '';
  const rows = [...filteredRows].sort((a,b)=>(toNum(b.ehagi)||-1)-(toNum(a.ehagi)||-1)).slice(0,60);
  for (const r of rows){
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${fmt(r.ehagi_rank,0)}</td><td>${r.county_name || r.acs_name || r.county_fips}</td><td>${r.state || ''}</td><td>${fmt(r.ehagi)}</td><td>${fmt(r.climate_hazard_burden)}</td><td>${fmt(r.health_vulnerability)}</td><td>${r.priority_group || ''}</td>`;
    tr.addEventListener('click', () => showProfile(r));
    tbody.appendChild(tr);
  }
}

function updateScatter(){
  const rows = filteredRows.filter(r => toNum(r.climate_hazard_burden)!==null && toNum(r.adaptive_capacity_deficit)!==null);
  const trace = {
    x: rows.map(r => toNum(r.climate_hazard_burden)),
    y: rows.map(r => toNum(r.adaptive_capacity_deficit)),
    text: rows.map(r => countyLabel(r)),
    mode: 'markers',
    type: 'scatter',
    marker: {
      size: rows.map(r => Math.max(6, Math.min(24, 6 + (toNum(r.ehagi)||0)/8))),
      opacity: 0.72,
      color: rows.map(r => toNum(r.ehagi))
    },
    hovertemplate: '<b>%{text}</b><br>Hazard burden: %{x:.1f}<br>Adaptive capacity deficit: %{y:.1f}<extra></extra>'
  };
  const layout = {
    margin: { l: 60, r: 15, t: 10, b: 55 },
    xaxis: { title: 'Climate/natural hazard burden' },
    yaxis: { title: 'Adaptive capacity deficit' },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    showlegend: false
  };
  Plotly.react('scatter', [trace], layout, { displayModeBar: false, responsive: true });
}

function showProfile(r){
  const el = document.getElementById('countyProfile');
  document.getElementById('profileHint').textContent = r.priority_group || '';
  el.className = 'profile';
  el.innerHTML = `
    <h3>${countyLabel(r)}</h3>
    <span class="badge">${r.priority_group || 'County profile'}</span>
    <p>A county-level summary of environmental-health risk and adaptive capacity. This is descriptive, not causal.</p>
    <div class="profile-grid">
      <div class="profile-stat"><span>EHAGI</span><strong>${fmt(r.ehagi)}</strong></div>
      <div class="profile-stat"><span>National rank</span><strong>${fmt(r.ehagi_rank,0)}</strong></div>
      <div class="profile-stat"><span>Hazard burden</span><strong>${fmt(r.climate_hazard_burden)}</strong></div>
      <div class="profile-stat"><span>Pollution exposure</span><strong>${fmt(r.air_pollution_exposure)}</strong></div>
      <div class="profile-stat"><span>Health vulnerability</span><strong>${fmt(r.health_vulnerability)}</strong></div>
      <div class="profile-stat"><span>Adaptive deficit</span><strong>${fmt(r.adaptive_capacity_deficit)}</strong></div>
      <div class="profile-stat"><span>Max AQI</span><strong>${fmt(r.max_aqi)}</strong></div>
      <div class="profile-stat"><span>PM2.5 mean</span><strong>${fmt(r.pm25_mean)}</strong></div>
      <div class="profile-stat"><span>Asthma prevalence</span><strong>${fmt(r.asthma_prev)}%</strong></div>
      <div class="profile-stat"><span>Poverty rate</span><strong>${fmt(r.poverty_rate)}%</strong></div>
    </div>
  `;
}

loadData().catch(err => {
  console.error(err);
  document.getElementById('lastUpdated').textContent = 'Dashboard could not load data. Check docs/data files.';
});
