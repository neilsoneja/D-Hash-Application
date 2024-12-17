<?php  
session_start(); 
$_SESSION['upload-message']="";
//$_SESSIOIN['cart']=array();
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
    <link rel="stylesheet" type="text/css" href="css/bootstrap/bootstrap-datepicker.min.css"> 
    <link rel="stylesheet" href="css/bootstrap/signin.css">
    <link rel="stylesheet" type="text/css" href="css/font-awesome.min.css">
    <link rel="stylesheet" type="text/css" href="/Kaye-Bakes-2/account/css/bootstrap/form-validation.css">
    <link rel="stylesheet" type="text/css" href="css/loader.css">
    <link rel="stylesheet" type="text/css" href="/Kaye-Bakes-2/account/css/list-groups.css">
    
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

<body class="text-center " >
<!--Loader-->
<div class="loader" id="loading"> 
  <div class="cssload-container" id="loading-image">
    <div class="cssload-circle"></div>
    <div class="cssload-circle"></div>
  </div>
</div>


  
  <main class="form-signin w-100 m-auto">

    <!--<img class="mb-4" src="/Kaye-Bakes-2/images/logo1.png" alt=""  height="57">-->
    <h1 class="sailyme mb-5 ground-color col s12">Welcome!</h1>

    <div id="login-page" class="row">
      <div class="container col s12 z-depth-4 card-panel">

        <form  id="form-control" method="post" action="datetime-confirm.php"  class="col s12 ">
        
          <div class="form-floating  mb-4">  
              <h5 class="bebas center">How do you want your order?</h5>
          </div>

          <!--
          <div class="row margin">
            <div class="input-field col s12">
              <label for="order_type" class="center-align"></label>
              <select name="order_type" id="order_type" type="text" data-error=".errorTxt2 ">
                <option value=""></option>
                <option value="menu">From the Menuers</option>
                <option value="custom">Customized</option>
              </select>
            </div>
          </div>
          -->



             
        
     



          <div class="form-floating mb-3">
            <div  class="input-group date" >
              
              <select name="order_type" id="order_type"  class="form-control" required>
                <option value="" disabled selected hidden>Order Type</option>
                <option value="menu">Choose from the Menu</option>
                <option value="custom">Customized Cake</option>
                </select>
                <span class="input-group-append">
                  <span class="input-group-text bg-white d-block">
                      <i class="fa fa-cupcake"></i>
                  </span>
              </span>
            </div>
            </div>


          <div class="form-floating mb-3">
          
                <div name="date_delivery" class="input-group date" id="datepicker">
                    <input type="text" class="form-control" name="date_delivery" id="date_delivery" placeholder="Delivery Date" required>
                    <span class="input-group-append">
                        <span class="input-group-text bg-white d-block">
                            <i class="fa fa-calendar"></i>
                        </span>
                    </span>
                </div>
            </div>
        

          
          <button class="ground-dark w-100 btn btn-lg btn-secondary mb-3" type="submit">Order</button>

           
          
          </form>




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
  

  <!--bootstrap-->
  <script type="text/javascript" src="js/bootstrap/bootstrap-datepicker.min.js"></script>
 
   <!--custom-->
   <script type="text/javascript" src="js/own-script.js"></script>
    
   <!--loader-->
    <script src="js/functions.js" type="text/javascript"></script>
 
    
    <script type="text/javascript">
        var date = new Date();
        date.setDate(date.getDate());

        $('#datepicker').datepicker({ 
         startDate: date
        });

        $(window).on('load', function () {
          $('#loading').fadeOut("slow");
          }) 

    </script>
  

</body>
</html>
<?php

?>