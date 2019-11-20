$("#file-picker2").change(function() {
    console.log('fdfd')
    var input = document.getElementById('file-picker2');
    for (var i = 0; i < input.files.length; i++) {
        var ext = input.files[i].name.substring(input.files[i].name.lastIndexOf('.') + 1).toLowerCase()
        if ((ext == 'jpg') || (ext == 'png') || (ext == 'jpeg')) {
            $("#msg").text("Files supported")
        } else {
            $("#msg").text("Files NOT supported")
            document.getElementById("file-picker2").value = "";
        }
    }
});

function readURL2(input) {
    if (input.files && input.files[0]) {

        var reader = new FileReader();

        reader.onload = function(e) {
            $('.image-upload-wrap2').hide();

            $('.file-upload-image2').attr('src', e.target.result);
            $('.file-upload-content2').show();

            //   $('.image-title').html(input.files[0].name);
        };

        reader.readAsDataURL(input.files[0]);

    }
}


$('.image-upload-wrap2').bind('dragover', function() {
    $('.image-upload-wrap2').addClass('image-dropping2');
});
$('.image-upload-wrap2').bind('dragleave', function() {
    $('.image-upload-wrap2').removeClass('image-dropping2');
});


function updateScore() {
    console.log('je me lance')
    var list = document.getElementsByClassName('score');
    for (var i = 0; i < list.length; i++) {
        if (list[i].textContent < 0.5) {
            list[i].style.backgroundColor = '#7CB342';
        } else if (list[i].textContent < 1) {
            list[i].style.backgroundColor = '#FFEB3B';
        } else {
            list[i].style.backgroundColor = "#E64A19";
        }
    }
}