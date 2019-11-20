$("#file-picker").change(function() {
    console.log('fdfd')
    var input = document.getElementById('file-picker');
    for (var i = 0; i < input.files.length; i++) {
        var ext = input.files[i].name.substring(input.files[i].name.lastIndexOf('.') + 1).toLowerCase()
        if ((ext == 'jpg') || (ext == 'png') || (ext == 'jpeg')) {
            $("#msg").text("Files supported")
        } else {
            $("#msg").text("Files NOT supported")
            document.getElementById("file-picker").value = "";
        }
    }
});

function readURL(input) {
    if (input.files && input.files[0]) {

        var reader = new FileReader();

        reader.onload = function(e) {
            $('.image-upload-wrap').hide();

            $('.file-upload-image').attr('src', e.target.result);
            $('.file-upload-content').show();

            //   $('.image-title').html(input.files[0].name);
        };

        reader.readAsDataURL(input.files[0]);

    }
}



$('.image-upload-wrap').bind('dragover', function() {
    $('.image-upload-wrap').addClass('image-dropping');
});
$('.image-upload-wrap').bind('dragleave', function() {
    $('.image-upload-wrap').removeClass('image-dropping');
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

function readURL2(input) {
    if (input.files && input.files[0]) {

        var reader = new FileReader();

        reader.onload = function(e) {
            $('.image-upload-wrap').hide();

            $('.file-upload-image').attr('src', e.target.result);
            $('.file-upload-content').show();

            //   $('.image-title').html(input.files[0].name);
        };

        reader.readAsDataURL(input.files[0]);

    }
}