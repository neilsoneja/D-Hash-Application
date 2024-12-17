<?php
session_start();
$dbservername = "localhost";
$dbUsername = "root";
$dbPassword = "";
$dbName = "kaye-bakes";
$conn = mysqli_connect($dbservername,$dbUsername,$dbPassword,$dbName);

$order_id=$_POST['order_id'];



$lastName=$_POST['lastName'];
$firstName= $_POST['firstName'];
$customer_id = $_POST['customer_id'];           
$address_main = $_POST['address_main'];
$address_specific = $_POST['address_specific'];
$order_date = $_POST['order_date'];
$payment = $_POST['payment'];
$order_type = $_POST['order_type'];
$date_delivery = $_POST['date_delivery'];
$time_delivery = $_POST['time_delivery'];
$delivery_options = $_POST['delivery_options'];
$shipping_mode = $_POST['shipping_mode'];
$shipping_fee = $_POST['shipping_fee'];
$total = $_POST['total'];
$dedications = $_POST['dedications'];
$requests_details = $_POST['requests_details'];

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
    <link rel="stylesheet" type="text/css" href="/Kaye-Bakes-2/account/css/bootstrap/dashboard.css">
    


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
     
<header class="navbar navbar-dark sticky-top bg-dark flex-md-nowrap p-0 shadow">
  <a class="navbar-brand col-md-3 col-lg-2 me-0 px-3 fs-6" href="#">Kaye Bakes</a>
  <button class="navbar-toggler position-absolute d-md-none collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#sidebarMenu" aria-controls="sidebarMenu" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
 <input class="form-control form-control-dark w-100 rounded-0 border-0" type="text" placeholder="Search" aria-label="Search">
    
</header>

<div class="container-fluid">
  <div class="row">
    <nav id="sidebarMenu" class="col-md-3 col-lg-2 d-md-block bg-light sidebar collapse">
      <div class="position-sticky pt-3">
        <ul class="nav flex-column">
          <li class="nav-item">
            <a class="nav-link" aria-current="page" href="/Kaye-Bakes-2/account/dashboard/">
              <span data-feather="home" class="align-text-bottom"></span>
              Dashboard
            </a>
          </li>


        <!--
          <li class="nav-item">
            <a class="nav-link" href="#">
              <span data-feather="file" class="align-text-bottom"></span>
              Orders
            </a>
          </li>

          <li class="nav-item">
            <a class="nav-link" href="#">
              <span data-feather="users" class="align-text-bottom"></span>
              Customers
            </a>
          </li>
    -->

          <li class="nav-item">
            <a class="nav-link  active" href="#" disabled>
              <span data-feather="shopping-cart" class="align-text-bottom"></span>
              Products
            </a>
          </li>

  
      </div>
    </nav>

    <main class="col-md-9 ms-sm-auto col-lg-10 px-md-4">
      <div class="d-flex justify-content-between flex-wrap flex-md-nowrap align-items-center pt-3 pb-2 mb-3 border-bottom">
        <h1 class="h2">Dashboard</h1>
        <div class="btn-toolbar mb-2 mb-md-0">
          <div class="btn-group me-2">
            <button type="button" class="btn btn-sm btn-outline-secondary">Share</button>
            <button type="button" class="btn btn-sm btn-outline-secondary">Export</button>
          </div>
          <button type="button" class="btn btn-sm btn-outline-secondary dropdown-toggle">
            <span data-feather="calendar" class="align-text-bottom"></span>
            This week
          </button>
        </div>
      </div>

      <!--top details-->
      <h4>Cart</h4>
      <div class="row">
          <div class="col-md-auto">
            <p class="display-4"><?php echo  ucwords($firstName)." ".ucwords($lastName);?></p> 
            
          </div>
      </div>
      <div class="row">    
          <div class="col-md-auto">
            <p class="fs-3"><?php echo ucwords($address_specific).", ".ucwords($address_main);?></p>
          </div>
      </div>
      <div class="row">
          <div class="col-md-auto">
            <p class="fs-5"><?php echo "To be delivered on ".$date_delivery." ".date('h:i a', strtotime($time_delivery));?></p>
          </div>
      </div>

      <!--cart and side details-->
     
            <?php
              $sql = "SELECT * FROM cart WHERE order_id = ".strval($order_id)."";
              $cart = mysqli_query($conn, $sql);

               while ($item = mysqli_fetch_assoc($cart) ){

                  
                  $sql = "SELECT * FROM products WHERE product_id = ".strval($item['product_id'])."";
                  $product = mysqli_query($conn, $sql);
                  $qproduct = mysqli_fetch_assoc($product);
                  if ($qproduct['customized']==1){

                  ?>
                  <hr>
                  <div class="row">
                    <div class="col-md-auto">
                      <img src="<?php  echo $qproduct['image_url']?>" style="height: 300px"> <br> 
                      <?php echo $qproduct['product_name'];?>
                    </div>  
                    <div class="col">
                      <div class="input-group">
                        <span class="input-group-text mb-3 ">Price PHP</span>
                        <input type="text" class="form-control mb-3 "  placeholder="<?php echo $qproduct['price'];?>">

                        <span class="input-group-text mb-3 ms-3">Type</span>
                        <input type="text" class="form-control mb-3 "  placeholder="<?php echo $qproduct['cake_type'];?>">
                      
                        <span class="input-group-text mb-3 ms-3">Size</span>
                        <input type="text" class="form-control mb-3 "  placeholder="<?php echo $qproduct['cake_size'];?>">
                      </div>
                      <div class="input-group">
                        <span class="input-group-text ">Flavor</span>
                        <input type="text" class="form-control "  placeholder="<?php echo $qproduct['cake_flavor'];?>">

                        <span class="input-group-text ms-3">Icing</span>
                        <input type="text" class="form-control "  placeholder="<?php echo $qproduct['icing_flavor'];?>">

                        <span class="input-group-text ms-3">Quantity</span>
                        <input type="text" class="form-control "  placeholder="<?php echo $item['quantity'];?>"><br>
                      </div>
                      <label for="dedications" class="form-label mt-3">Dedications</label>
                      <textarea class="form-control" rows="4" name="dedications" placeholder="<?php echo $dedications; ?>"></textarea>
                      <label for="requests_details" class="form-label">Requests</label>
                      <textarea class="form-control" rows="4" name="requests_details" placeholder="<?php echo $requests_details; ?>"></textarea>
                    </div>

                      

           
                  <?php
    
                }
                else{
                    ?>
                     <hr>
                  <div class="row">
                    <div class="col-md-auto">
                      <img src="<?php  echo "/Kaye-Bakes-2/account/product_images/".$qproduct['image_url']?>" style="height: 300px"> <br> 
                      <?php echo $qproduct['product_name'];?>
                    </div>  
                    <div class="col">
                      <div class="input-group">
                        <span class="input-group-text mb-3 ">Price PHP</span>
                        <input type="text" class="form-control mb-3 "  placeholder="<?php echo $qproduct['price'];?>">

                      </div>
                      <div class="input-group">
                        
                        <span class="input-group-text ">Quantity</span>
                        <input type="text" class="form-control "  placeholder="<?php echo $item['quantity'];?>"><br>
                      </div>
                      <label for="dedications" class="form-label mt-3">Dedications</label>
                      <textarea class="form-control" rows="4" name="dedications" placeholder="<?php echo $dedications; ?>"></textarea>
                      <label for="requests_details" class="form-label">Requests</label>
                      <textarea class="form-control" rows="4" name="requests_details" placeholder="<?php echo $requests_details; ?>"></textarea>
                    </div>

                      

                    <?php    
                }
               }
              
            ?>
             
          </div>
      </div>

      <!--bottom details-->
      <div class="row"></div>
      <div class="table-responsive">


 
            



      </div>
    </main>
  </div>
</div>


    <!--js-->
  <!--bootstrap-->
  <script type="text/javascript" src="/Kaye-Bakes-2/account/js/bootstrap/feather.min.js"></script>
  <script type="text/javascript" src="/Kaye-Bakes-2/account/js/bootstrap/bootstrap.bundle.min.js"></script>  
  <script type="text/javascript" src="/Kaye-Bakes-2/account/js/bootstrap/dashboard.js"></script> 
  <script type="text/javascript" src="/Kaye-Bakes-2/account/js/bootstrap/Chart.min.js"></script>
   

<!-- jQuery Library 
<script type="text/javascript" src="/Kaye-Bakes-2/account/js/jquery-3.4.1.min.js"></script>-->

<!--bootstrap
<script type="text/javascript" src="/Kaye-Bakes-2/account/js/bootstrap/bootstrap-datepicker.min.js"></script>-->

<!--custom-->
<script type="text/javascript" src="/Kaye-Bakes-2/account/js/own-script.js"></script>


</body>
</html>
