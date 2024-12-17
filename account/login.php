<?php  
session_start(); 
if(isset($_SESSION['admin_sid']) || isset($_SESSION['customer_sid']))
{
	header("location:index.php");
}
else{
?>
<!DOCTYPE html>
<html lang="en">

<head>
  <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1.0, user-scalable=no">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <meta name="msapplication-tap-highlight" content="no">
  <title>Login</title>

  <!-- Favicons-->
  <link rel="icon" href="images/favicon/favicon-32x32.png" sizes="32x32">
  <!-- Favicons-->
  <link rel="apple-touch-icon-precomposed" href="images/favicon/apple-touch-icon-152x152.png">
  <!-- For iPhone -->
  <meta name="msapplication-TileColor" content="#00bcd4">
  <meta name="msapplication-TileImage" content="images/favicon/mstile-144x144.png">
  <!-- For Windows Phone -->


  <!-- CORE CSS-->
  
  <link href="css/materialize.min.css" type="text/css" rel="stylesheet" media="screen,projection">
  <link href="css/style.min.css" type="text/css" rel="stylesheet" media="screen,projection">
    <!-- Custome CSS-->    
    <link href="css/custom/custom.min.css" type="text/css" rel="stylesheet" media="screen,projection">
  <link href="css/layouts/page-center.css" type="text/css" rel="stylesheet" media="screen,projection">

  <!-- INCLUDED PLUGIN CSS ON THIS PAGE -->
  <link href="js/plugins/perfect-scrollbar/perfect-scrollbar.css" type="text/css" rel="stylesheet" media="screen,projection">
  
</head>

<body class="cyan">
  <!-- Start Page Loading -->
  <div id="loader-wrapper">
      <div id="loader"></div>        
      <div class="loader-section section-left"></div>
      <div class="loader-section section-right"></div>
  </div>
  <!-- End Page Loading -->



  <div id="login-page" class="row">
    <div class="col s12 z-depth-4 card-panel">
      <form method="post" action="routers/router.php" class="login-form" id="form">
        <div class="row">
          <div class="input-field col s12 center">
            <p class="center login-form-text">Login for Food Ordering System</p>
          </div>
        </div>
        <div class="row margin">
          <div class="input-field col s12">
            <i class="mdi-social-person-outline prefix"></i>
            <input name="username" id="username" type="text">
            <label for="username" class="center-align">Username</label>
          </div>
        </div>
        <div class="row margin">
          <div class="input-field col s12">
            <i class="mdi-action-lock-outline prefix"></i>
            <input name="password" id="password" type="password">
            <label for="password">Password</label>
          </div>
        </div>
        <div class="row">
			<a href="javascript:void(0);" onclick="document.getElementById('form').submit();" class="btn waves-effect waves-light col s12">Login</a>
          </div>
		  		<div class="row">
          <div class="input-field col s6 m6 l6">
            <p class="margin medium-small"><a href="register.php">Register Now!</a></p>
          </div>         
        </div>
        </div>


      </form>
    </div>
  </div>



  <!-- ================================================
    Scripts
    ================================================ -->

  <!-- jQuery Library -->
  <script type="text/javascript" src="js/plugins/jquery-1.11.2.min.js"></script>
  <!--materialize js-->
  <script type="text/javascript" src="js/materialize.min.js"></script>
  <!--scrollbar-->
  <script type="text/javascript" src="js/plugins/perfect-scrollbar/perfect-scrollbar.min.js"></script>

      <!--plugins.js - Some Specific JS codes for Plugin Settings-->
    <script type="text/javascript" src="js/plugins.min.js"></script>
    <!--custom-script.js - Add your own theme custom JS-->
    <script type="text/javascript" src="js/custom-script.js"></script>

</body>
</html>
<?php
}
?>


<!---

<div class="menu">
        <form method="post" action="routers/add-to-cart.php?product_id=<?php$row['product_id'];?>">
          <div class="product-container">
            <div class="card">
              /*
              <?php
              /*
                $sql = "SELECT * FROM products WHERE customized = 0";
                $result = mysqli_query($conn, $sql);
                $query_results = mysqli_fetch_array($result);
    
                if ($query_results > 0) {
                  while ($row = mysqli_fetch_assoc($result)) {
                    echo "<a href='product.php?title=".$row['product_name']."&image=".$row['image_url']."'><div class='product-box'>
                    <img src='product_images/".$row['image_url']."' >
                    <p hidden>".$row['product_id']."</p>
                    <h3>".$row['product_name']."</h3>
                    <p hidden>".$row['product_desc']."</p>
                    <p>Php ".number_format($row['price'])."</p>
                    <input type='hidden' name='product_name' value='$row['product_name']'>
                    <input type='hidden' name='product_desc' value='$row['product_desc']'>
                    <input type='hidden' name='price' value='$row['price']'>
                    <input type='hidden' name='quantity' value='$row['quantity']'>
                    <button type='submit' name='add-to-cart'>Add to Cart</button>
                  </div></a>";
                  }
                }
                */
              ?>

--->

<?php
                $sql = "SELECT * FROM products WHERE customized = 0";
                $result = mysqli_query($conn, $sql);
                $query_results = mysqli_fetch_array($result);
    
                if ($query_results > 0) {
                  while ($row = mysqli_fetch_assoc($result)) {?>

                    <?php
                    echo "<a href='product.php?title=".$row['product_name']."&image=".$row['image_url']."'>"
                    ?>

                    <form method="post" action="routers/add-to-cart.php?product_id=<?=$row['product_id']?>">
                    <img src="product_images/<?=$row['image_url']?>" style='height: 250px'>
                    <p><?=$row['product_name'];?></p>
                    <p hidden><?=$row['product_desc'];?></p>
                    <p>Php  <?=number_format($row['price']);?></p>

                    <input type="hidden" name="image_url" value="<?= $row['image_url']?>">
                    <input type="hidden" name="product_name" value="<?= $row['product_name']?>">
                    <input type="hidden" name="product_desc" value="<?= $row['product_desc']?>">
                    <input type="hidden" name="price" value="<?= $row['price']?>">
                    <input type="number" name="quantity" value="1" hidden>
                    <input type="submit" name="add-to-cart" value="Add To Cart">
                    </form>
                    <?php }
                }
                

              ?>