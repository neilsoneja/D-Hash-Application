<?php
session_start();

$_SESSION['upload-message']="";

?>



<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>customize</title>

    <link rel="stylesheet" type="text/css"  media="screen,projection" href="/Kaye-Bakes-2/account/css/new.css">
    <link rel="stylesheet" type="text/css" href="/Kaye-Bakes-2/account/css/bootstrap/bootstrap.min.css">
    <link rel="stylesheet" type="text/css" href="/Kaye-Bakes-2/account/css/font-awesome.min.css">
    <link rel="stylesheet" type="text/css" href="/Kaye-Bakes-2/account/css/loader.css">

    <style>
      .bd-placeholder-img {
        font-size: 1.125rem;
        text-anchor: middle;
        -webkit-user-select: none;
        -moz-user-select: none;
        user-select: none;
      }

      @media (min-width: 768px) {
        .bd-placeholder-img-lg {
          font-size: 3.5rem;
        }
      }

      .b-example-divider {
        height: 3rem;
        background-color: rgba(0, 0, 0, .1);
        border: solid rgba(0, 0, 0, .15);
        border-width: 1px 0;
        box-shadow: inset 0 .5em 1.5em rgba(0, 0, 0, .1), inset 0 .125em .5em rgba(0, 0, 0, .15);
      }

      .b-example-vr {
        flex-shrink: 0;
        width: 1.5rem;
        height: 100vh;
      }

      .bi {
        vertical-align: -.125em;
        fill: currentColor;
      }

      .nav-scroller {
        position: relative;
        z-index: 2;
        height: 2.75rem;
        overflow-y: hidden;
      }

      .nav-scroller .nav {
        display: flex;
        flex-wrap: nowrap;
        padding-bottom: 1rem;
        margin-top: -1px;
        overflow-x: auto;
        text-align: center;
        white-space: nowrap;
        -webkit-overflow-scrolling: touch;
      }
    </style>

    
</head>
<body>

    <!--Loader-->
    <div class="loader" id="loading"> 
      <div class="cssload-container" id="loading-image">
        <div class="cssload-circle"></div>
        <div class="cssload-circle"></div>
      </div>
    </div>

<?php


$target_dir = "uploads/";
$target_file = $target_dir . basename($_FILES["fileToUpload"]["name"]);
$uploadOk = 1;
$imageFileType = strtolower(pathinfo($target_file,PATHINFO_EXTENSION));

// Check if image file is a actual image or fake image
  if(isset($_POST["submit"])) {
    $check = getimagesize($_FILES["fileToUpload"]["tmp_name"]);
    if($check !== false) {
      //echo "File is an image - " . $check["mime"] . ".";
      $uploadOk = 1;
    } else {
      $_SESSION['upload-message']= "File is not an image.";
      $uploadOk = 0;
      echo "<script> location.href='/Kaye-Bakes-2/account/customize-order-new.php'; </script>";
      exit;
    }
  }

// Check if file already exists
  if (file_exists($target_file)) {
    $_SESSION['upload-message']= "Sorry, file already exists.";
    $uploadOk = 0;
    echo "<script> location.href='/Kaye-Bakes-2/account/customize-order-new.php'; </script>";
    exit;
  }

// Check file size
  if ($_FILES["fileToUpload"]["size"] > 5000000) {
    $_SESSION['upload-message']= "Sorry, your file is too large.";
    $uploadOk = 0;
    echo "<script> location.href='/Kaye-Bakes-2/account/customize-order-new.php'; </script>";
    exit;
  }

// Allow certain file formats
  if($imageFileType != "jpg" && $imageFileType != "png" && $imageFileType != "jpeg" ) {
    $_SESSION['upload-message']= "Sorry, only JPG, JPEG, & PNG files are allowed.";
    $uploadOk = 0;
    echo "<script> location.href='/Kaye-Bakes-2/account/customize-order-new.php'; </script>";
    exit;
  }

// Check if $uploadOk is set to 0 by an error
if ($uploadOk == 1) {

    //renaming
    $temp = explode(".", $_FILES["fileToUpload"]["name"]);

    if (isset($_SESSION['uploadNumber'])){
      
      $_SESSION['uploadNumber']+=1;
    }
    else {
      $_SESSION['uploadNumber']=1;
    }
    /*$newfilename= random_bytes(20);
    $newfilename = $newfilename . '.' . end($temp);*/
    $newfilename = strval($_SESSION['uploadNumber']) . '.' . end($temp);
  
    //setting name as session var for search  
    $_SESSION['query-filename'] = $newfilename;

    
    
    //Uploading
    if (move_uploaded_file($_FILES["fileToUpload"]["tmp_name"], "uploads/".$newfilename)) {    
     
      echo "<script> location.href='/Kaye-Bakes-2/account/Dhash/search-results.php'; </script>";
      exit;  
      }
    else {
      $_SESSION['upload-message']= "Sorry, there was an error uploading your file.<br>Please try again.";}
      echo "<script> location.href='/Kaye-Bakes-2/account/customize-order-new.php'; </script>";
      exit;


  
}

else {
  $_SESSION['upload-message']= "Sorry, an error occured. Your file was not uploaded. <br> Please try again.";
}?>


<!--js-->
  <!--bootstrap-->
  <script type="text/javascript" src="/Kaye-Bakes-2/account/js/bootstrap/bootstrap.bundle.min.js"></script>  

  <!-- jQuery Library -->
  <script type="text/javascript" src="/Kaye-Bakes-2/account/js/jquery-3.4.1.min.js"></script>

  <!--bootstrap-->
  <script type="text/javascript" src="/Kaye-Bakes-2/account/js/bootstrap/bootstrap-datepicker.min.js"></script>

  <!--custom-->
  <script type="text/javascript" src="/Kaye-Bakes-2/account/js/own-script.js"></script>

  <!--loader-->
  <script src="js/functions.js" type="text/javascript"></script>
<script>  
  function preview() {
      frame.src = URL.createObjectURL(event.target.files[0]);
  }
  function clearImage() {
      document.getElementById('formFile').value = null;
      frame.src = "";
  }


  $(window).on('load', function () {
    $('#loading').fadeOut("slow");
  }) 
</script>    

</body>
</html>