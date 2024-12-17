<?php
session_start();
$dbservername = "localhost";
$dbUsername = "root";
$dbPassword = "";
$dbName = "kaye-bakes";
$conn = mysqli_connect($dbservername,$dbUsername,$dbPassword,$dbName);

//database initialization
/*
if (!isset($_SESSION['dashcart'])){
  foreach ($_SESSION['orders'] as $key=>$value){
    $sql = "SELECT * FROM cart WHERE order_id = ".$value['order_id']."";
    $dashcart = mysqli_query($conn, $sql);
    while ($row = mysqli_fetch_assoc($dashcart)){      
      $dashcart_array = array (
        "item_id" => $row['item_id'],
        "product_id" => $row['product_id'],
        "order_id" => $row['order_id'],
        "quantity" => $row['quantity'],
        "price" => $row['price']
      );
      $_SESSION['dashcart'][$row['product_id']] = $dashcart_array;
  }}}

if (!isset($_SESSION['products'])){
  foreach ($_SESSION['dashcart'] as $key=>$value){
    $sql = "SELECT * FROM products WHERE product_id = ".$value['product_id']."";
    $products = mysqli_query($conn, $sql);
    while ($row = mysqli_fetch_assoc($products)){
      $products_array = array (
        "product_id" => $row['product_id'],
        "product_name" => $row['product_name'],
        "product_desc" => $row['product_desc'],
        "price" => $row['price'],
        "cake_type" => $row['cake_type'],
        "cake_size" => $row['cake_size'],
        "cake_flavor" => $row['cake_flavor'],
        "icing_flavor" => $row['icing_flavor'],
        "customized" => $row['customized'],
        "image_url" => $row['image_url']
      );
      $_SESSION['products'][$row['product_id']] = $products_array;
  }}}
*/
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
            <a class="nav-link active" aria-current="page" href="#">
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
            <a class="nav-link" href="#">
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

      
      <h4>Pending Orders</h4>
      <div class="table-responsive">
        <table class="table table-striped table-sm">
          <thead>
            <tr>
              <th scope='col'>Order ID</th>
              <th scope='col'>Customer</th>
              <th scope='col'>Address</th>
              <th scope='col'>Order Date</th>
              <th scope='col'>Delivery Date</th>
              <th scope='col'>Delivery Time</th>
            </tr>
          </thead>

          <tbody>
             <?php


              $sql = "SELECT * FROM orders WHERE order_status = 'pending' ORDER BY order_id DESC";
              $orders = mysqli_query($conn, $sql);

              while ($row = mysqli_fetch_assoc($orders)){

                
                ?>
                  <form action="orders/index.php" method="post">
                    <tr>
                      <th>
                        <button name="" type="submit" class="btn btn-link"> 
                          <?php echo $row['order_id'];?>
                        </button>
                      </th>
                      <td><?php 
                        $sql = "SELECT lastName, firstName FROM customers WHERE customer_id = ".strval($row['customer_id'])."";
                        $names = mysqli_query($conn, $sql);
                        $query_name = mysqli_fetch_array($names);
                        echo ucwords($query_name['firstName'])." ".ucwords($query_name['lastName']);
                      ?></td>
                      <td><?php echo ucwords($row['address_main']);?></td>
                      <td><?php echo $row['order_date'];?></td>
                      <td><?php echo $row['date_delivery'];?></td>
                      <td><?php echo $row['time_delivery'];?></td>
                    </tr>

                    <input type="hidden" name="order_id" value="<?= $row['order_id']?>">
                    <input type="hidden" name="lastName" value="<?= $query_name['lastName']?>">
                    <input type="hidden" name="firstName" value="<?= $query_name['firstName']?>">
                    <input type="hidden" name="customer_id" value="<?= $row['customer_id']?>">                    
                    <input type="hidden" name="address_main" value="<?= $row['address_main']?>">
                    <input type="hidden" name="address_specific" value="<?= $row['address_specific']?>">
                    <input type="hidden" name="order_date" value="<?= $row['order_date']?>">
                    <input type="hidden" name="payment" value="<?= $row['payment']?>">
                    <input type="hidden" name="order_type" value="<?= $row['order_type']?>">
                    <input type="hidden" name="date_delivery" value="<?= $row['date_delivery']?>">
                    <input type="hidden" name="time_delivery" value="<?= $row['time_delivery']?>">
                    <input type="hidden" name="delivery_options" value="<?= $row['delivery_options']?>">
                    <input type="hidden" name="shipping_mode" value="<?= $row['shipping_mode']?>">
                    <input type="hidden" name="shipping_fee" value="<?= $row['shipping_fee']?>">
                    <input type="hidden" name="total" value="<?= $row['total']?>">
                    <input type="hidden" name="dedications" value="<?= $row['dedications']?>">
                    <input type="hidden" name="requests_details" value="<?= $row['requests_details']?>">

                  </form>
                <?php
              }
             
             

             ?>

 
            


          </tbody>
        </table>
      </div>





      <hr>
      <h4>Customers</h4>
      <div class="table-responsive">
        <table class="table table-striped table-sm">
          <thead>
            <tr>
              <th scope='col'>Name</th>
              <th scope='col'>Address</th>
              <th scope='col'>Email</th>
              <th scope='col'>Mobile</th>
            </tr>
          </thead>

          <tbody>
             <?php
                 
              $sql = "SELECT * FROM customers ORDER BY customer_id DESC" ;
              $customers = mysqli_query($conn, $sql);
                while ($row = mysqli_fetch_array($customers)){  
                ?>
                  <form action="orders/" method="post">
                    <tr>
                      <th>
                        <button type="submit" class="btn btn-link"> 
                          <?php echo $row['firstName']." ".$row['lastName'];?>
                        </button>
                      </th>
                      <td><?php echo $row['address_specific'].", ".$row['address_main'];?></td>
                      <td><?php echo $row['email'];?></td>
                      <td><?php echo $row['mobile'];?></td>
                    </tr>

                  </form>
                <?php
              }
             ?>
          </tbody>
        </table>
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
