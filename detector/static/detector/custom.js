 // ---------Autogrow textarea-----------
  function growTextarea (i,elem) {
    var elem = $(elem);
    var resizeTextarea = function( elem ) {
        var scrollLeft = window.pageXOffset || (document.documentElement || document.body.parentNode || document.body).scrollLeft;
        var scrollTop  = window.pageYOffset || (document.documentElement || document.body.parentNode || document.body).scrollTop;  
        elem.css('height', 'auto').css('height', elem.prop('scrollHeight') );
          window.scrollTo(scrollLeft, scrollTop);
      };
      elem.on('input', function() {
        resizeTextarea( $(this) );
      });
      resizeTextarea( $(elem) );
  }
  
  $('.jTextarea').each(growTextarea);
  