// to get current year
function getYear() {
    var currentDate = new Date();
    var currentYear = currentDate.getFullYear();
    var yearNode = document.querySelector("#displayYear");
    if (yearNode) {
        yearNode.innerHTML = currentYear;
    }
}

getYear();

(function loadThemeControls() {
    if (document.querySelector('script[src="/static/js/theme.js"]')) return;
    var script = document.createElement("script");
    script.src = "/static/js/theme.js";
    script.defer = true;
    document.head.appendChild(script);
})();

//  owl carousel script
if (window.jQuery && $(".owl-carousel").length) {
    $(".owl-carousel").owlCarousel({
        loop: true,
        margin: 20,
        nav: true,
        navText: [],
        autoplay: true,
        autoplayHoverPause: true,
        responsive: {
            0: {
                items: 1
            },
            1000: {
                items: 2
            }
        }
    });
}

//    end owl carousel script 



/** google_map js **/
function myMap() {
    var mapProp = {
        center: new google.maps.LatLng(40.712775, -74.005973),
        zoom: 18,
    };
    var map = new google.maps.Map(document.getElementById("googleMap"), mapProp);
}
