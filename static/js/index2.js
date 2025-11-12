  var mousePressed = false;
    var lastX, lastY;
    var ctx;

    function init() {
      canvas = document.getElementById('myCanvas');
      ctx = canvas.getContext('2d');
      ctx.fillStyle = 'white';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      $('#myCanvas').mousedown(function (e) {
        mousePressed = true;
        draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, false);
      });
      $('#myCanvas').mousemove(function (e) {
        if (mousePressed) {
          draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, true);
        }
      });
      $('#myCanvas').mouseup(function () { mousePressed = false; });
      $('#myCanvas').mouseleave(function () { mousePressed = false; });
    }

    function clearCanvas() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = 'white';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      document.getElementById("result").innerHTML = "Canvas cleared 🧼";
    }

    function draw(x, y, isDown) {
      if (isDown) {
        ctx.beginPath();
        ctx.strokeStyle = $('#selColor').val();
        ctx.lineWidth = $('#selWidth').val();
        ctx.lineJoin = 'round';
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.closePath();
        ctx.stroke();
      }
      lastX = x;
      lastY = y;
    }

    function postImage() {
      var img = document.getElementById("myCanvas").toDataURL("image/png");
      img = img.replace(/^data:image\/(png|jpg);base64,/, "");
      document.getElementById("result").innerHTML = "⏳ Recognizing...";

      $.ajax({
        type: 'POST',
        url: '/recognize',
        data: JSON.stringify({ data: img }),
        contentType: 'application/json;charset=UTF-8',
        dataType: 'json',
        success: function (response) {
          document.getElementById("result").innerHTML =
            "✅ Recognized character: <b>" + response.prediction + "</b>";
        },
        error: function () {
          document.getElementById("result").innerHTML =
            "❌ Error recognizing image.";
        }
      });
    }