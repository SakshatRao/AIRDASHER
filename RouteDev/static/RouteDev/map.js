var map = L.map('map').setView([22, 85], 5);

L.tileLayer('http://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png', {
    maxZoom: 5,
    className: 'map-tiles',
    attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
}).addTo(map);

var circle = L.circle([12.994341981037747, 80.17159635853399], {
    color: 'black',
    fillColor: 'green',
    fillOpacity: 0.5,
    radius: 50000
}).addTo(map);
circle.bindPopup("<b>Chennai</b><br>MAA");

var circle = L.circle([19.099282513860523, 72.87502185672862], {
    color: 'black',
    fillColor: 'green',
    fillOpacity: 0.5,
    radius: 50000
}).addTo(map);
circle.bindPopup("<b>Mumbai</b><br>BOM");

var polylinePoints = [
    [19.099282513860523, 72.87502185672862],
    [12.994341981037747, 80.17159635853399],
    [17.240550198145755, 78.42981425116706]
];            

var polyline = L.polyline(polylinePoints).addTo(map);