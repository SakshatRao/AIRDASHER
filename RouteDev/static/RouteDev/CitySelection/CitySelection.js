var modals = []
var cities = []
var closes = []
var open_shortcuts = []

for (let i = 1; i <= 5; i++) {
    var modal_name = "CityModal" + i.toString();
    var modal = document.getElementById(modal_name);
    modals.push(modal)
    var city_name = "City" + i.toString();
    var city = document.getElementById(city_name);
    cities.push(city)
    var close_name = "close" + i.toString();
    var close = document.getElementById(close_name);
    closes.push(close)

    cities[i-1].onclick = function() {
        modals[i-1].style.display = 'block';
    }
    closes[i-1].onclick = function() {
        modals[i-1].style.display = 'none';
    }
}

window.onclick = function(event) {
    if (event.target.classList.contains('CityModal')) {
        for (let i = 1; i <= 5; i++) {
            modals[i-1].style.display = 'none';
        }
    }
}

function show_node_link(idx) {
    modals[idx].style.display = 'block';
}