var modals = []
var options = []
var closes = []

for (let i = 1; i <= 5; i++) {
    var modal_name = "OptionModal" + i.toString();
    var modal = document.getElementById(modal_name);
    modals.push(modal)
    var option_name = "Option" + i.toString();
    var option = document.getElementById(option_name);
    options.push(option)
    var close_name = "close" + i.toString();
    var close = document.getElementById(close_name);
    closes.push(close)

    options[i-1].onclick = function() {
        modals[i-1].style.display = 'block';
    }
    closes[i-1].onclick = function() {
        modals[i-1].style.display = 'none';
    }
}

window.onclick = function(event) {
    if (event.target.classList.contains('OptionModal')) {
        for (let i = 1; i <= 5; i++) {
            modals[i-1].style.display = 'none';
        }
    }
}

var slider_in = document.getElementById("price_in_slider");
var output_in = document.getElementById("selected_price_in");
output_in.innerHTML = slider_in.value;

slider_in.oninput = function() {
    output_in.innerHTML = this.value;
}