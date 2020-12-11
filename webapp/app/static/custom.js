$('ul.navbar-nav a').parent().removeClass('active');

var url = window.location;
// Will only work if string in href matches with location
$('ul.navbar-nav a[href="'+ url +'"]').parent().addClass('active');

// Will also work for relative and absolute hrefs
$('ul.navbar-nav a').filter(function() {
    return this.href == url;
}).parent().addClass('active');

$(document).ready(function () {
    bsCustomFileInput.init()
})