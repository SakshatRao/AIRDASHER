var modals = []
var routes = []
var closes = []

for (let i = 1; i <= 5; i++) {
    var modal_name = "RouteModal" + i.toString();
    var modal = document.getElementById(modal_name);
    modals.push(modal)
    var route_name = "Route" + i.toString();
    var route = document.getElementById(route_name);
    routes.push(route)
    var close_name = "close" + i.toString();
    var close = document.getElementById(close_name);
    closes.push(close)

    routes[i-1].onclick = function() {
        modals[i-1].style.display = 'block';
    }
    closes[i-1].onclick = function() {
        modals[i-1].style.display = 'none';
    }
}

window.onclick = function(event) {
    if (event.target.classList.contains('RouteModal')) {
        for (let i = 1; i <= 5; i++) {
            modals[i-1].style.display = 'none';
        }
    }
}

function show_route_link(idx) {
    modals[idx].style.display = 'block';
}