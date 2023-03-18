var MAPS_PLOT = MAPS_PLOT || (function(){
    var coords = [];
    var names = [];
    var airport_names = [];
    var is_hubs = [];
    var routes = [];
    var city_coords = [];
    var city_names = [];

    return {
        init_coords : function(coord) {
            coords.push(coord);
        },
        init_names : function(name) {
            names.push(name);
        },
        init_airport_names : function(airport_name) {
            airport_names.push(airport_name);
        },
        init_is_hubs : function(is_hub) {
            is_hubs.push(is_hub);
        },
        init_routes : function(route1, route2) {
            routes.push([route1, route2]);
        },
        init_city_coords : function(city_coord) {
            city_coords.push(city_coord)
        },
        init_city_names : function(city_name) {
            city_names.push(city_name)
        },
        plot_map : function(sample_network_name, show_network, show_node_link, show_route_link) {

            var tile_layer = L.tileLayer('http://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png', {
                maxZoom: 5,
                className: 'map-tiles',
                attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            });

            var network_layer = L.layerGroup();
            for (let i = 0; i < routes.length; i++) {
                var lat_lon1 = JSON.parse("[" + routes[i][0] + "]");
                var lat_lon2 = JSON.parse("[" + routes[i][1] + "]");
                var polylinePoints = [
                    [lat_lon1[0], lat_lon1[1]],
                    [lat_lon2[0], lat_lon2[1]]
                ];
                var line_info;
                if (show_route_link == 1) {
                    line_info = {
                        'color': 'white',
                        'weight': 5,
                        'opacity': 1
                    };
                }
                else {
                    line_info = {
                        'color': 'white',
                        'weight': 1.5,
                        'opacity': 1,
                        'dashArray': "4 1"
                    };
                }
                var polyline = L.polyline(polylinePoints, line_info);
                if (show_route_link == 1) {
                    polyline.bindPopup("<b>Route " + city_names[0] + "-" + names[i] + "</b><br><p onclick='show_route_link(" + i.toString() + ")'>Click on me!</p>");
                }
                else {
                    polyline.bindPopup("<b>Route " + city_names[0] + "-" + names[i] + "</b>");
                }
                network_layer.addLayer(polyline);
            }

            for (let i = 0; i < coords.length; i++) {
                var lat_lon = JSON.parse("[" + coords[i] + "]");
                var circle_info = {};
                if (is_hubs[i] == "False") {
                    circle_info = {
                        color: 'black',
                        fillColor: 'orange',
                        fillOpacity: 1,
                        radius: 30000
                    };
                }
                else {
                    circle_info = {
                        color: 'black',
                        fillColor: 'red',
                        fillOpacity: 1,
                        radius: 50000
                    };
                }
                var circle = L.circle([lat_lon[0], lat_lon[1]], circle_info);
                circle.bindTooltip("<b>" + names[i] + "</b><br>" + airport_names[i]);
                network_layer.addLayer(circle);
            }

            var new_cities_layer = L.layerGroup();
            for (let i = 0; i < city_coords.length; i++) {
                var lat_lon = JSON.parse("[" + city_coords[i] + "]");
                var circle_info = {};
                circle_info = {
                    fillColor: 'blue',
                    fillOpacity: 0.5,
                    radius: 70000
                };
                var circle = L.circle([lat_lon[0], lat_lon[1]], circle_info);
                if (show_node_link == 1) {
                    circle.bindPopup("<b>" + city_names[i] + "<br><p onclick='show_node_link(" + i.toString() + ")'>Click on me</p>");
                }
                else {
                    circle.bindPopup("<b>" + city_names[i]);
                }
                new_cities_layer.addLayer(circle);
            }

            var overlays = {}
            overlays["<b>" + sample_network_name + '</b> Network'] = network_layer;
            overlays['Highest-growth Cities'] = new_cities_layer;

            if (show_network == 1) {
                valid_layers = [tile_layer, new_cities_layer, network_layer]
            }
            else {
                valid_layers = [tile_layer, new_cities_layer]
            }
            var map = L.map('map', {
                center: [22, 85],
                zoom: 5,
                layers: valid_layers
            }).setView([22, 85], 5);

            var LayerControl = L.control.layers({}, overlays);
            LayerControl.addTo(map);
        }
    };
}());