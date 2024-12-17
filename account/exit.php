<?php
session_start();

$_SESSION['cart']=array();

?>

<!DOCTYPE html>
<html lang="en">
<head>

  <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1.0, user-scalable=no">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <meta name="msapplication-tap-highlight" content="no">
  <title>Order</title>

  <!-- Favicons-->
    <link rel="icon" href="images/favicon/favicon-32x32.png" sizes="32x32">
    <link rel="apple-touch-icon-precomposed" href="images/favicon/apple-touch-icon-152x152.png">
  <!-- For iPhone -->
    <meta name="msapplication-TileColor" content="#00bcd4">
    <meta name="msapplication-TileImage" content="images/favicon/mstile-144x144.png">
  <!-- For Windows Phone -->

  <!-- CORE CSS--> 
    <link rel="stylesheet" type="text/css"  media="screen,projection" href="css/new.css">
    <link rel="stylesheet" type="text/css" href="css/bootstrap/bootstrap.min.css">
    <link rel="stylesheet" href="css/bootstrap/signin.css">
    <link rel="stylesheet" type="text/css" href="css/font-awesome.min.css">
    
  <!-- CSS -->
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




<body class="text-center ">


<main class=" w-100 m-auto">
<div class="container ">
<div class=" row justify-content-center  py-4">
<!--<img class="mb-4" src="/Kaye-Bakes-2/images/logo1.png" alt=""  height="57">-->


<div id="login-page" class="col-md-6 bg-light rounded-3 shadow-sm pt-5 justify-content-center">
  <h1 class="bebas mb-3  col s12 display-3">Thank You!</h1>
  <div class="container col s12 z-depth-4 card-panel  ">

    
      <div class="form-floating  mb-4">  
        <div class="row">
          <h5 class="bebas center ground-color"><?php echo "Your   reference code is  ".$_SESSION['reference_code'].".<br>";?></h5>
        </div>
        <div class="row justify-content-center ">
          <div class="col-md-8 text-">
            <p class="justify ">Please wait to be contacted by Kaye Bakes to confirm your order. If you have any questions or concerns, feel free to contact us and present you reference code for faster transaction.</p>
          </div>
        </div>
        </div>


      <div class="form-floating  ">
     
      <a href="http://neilsoneja/Kaye-Bakes-2" class="ground-dark w-50 btn btn-sm btn-secondary mb-3">Back to Homepage</a>
       
  
     
      </div>
    </div>
  </div>
</div>
</div> 
</main>

  <!-- ================================================
    Scripts
    ================================================ -->


  <!--bootstrap-->
  <script type="text/javascript" src="js/bootstrap/bootstrap.bundle.min.js"></script>  

    <!-- jQuery Library -->
  <script type="text/javascript" src="js/jquery-3.4.1.min.js"></script>

  <!--custom-->
   <script type="text/javascript" src="js/own-script.js"></script>

    


</body>
</html>
