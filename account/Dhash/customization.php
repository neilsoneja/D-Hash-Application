<?php
session_start();
$baseName = $_POST["baseName"];
$cake_type=$_SESSION['cake_type'];
$url = $_POST["url"];

$basePath="/Kaye-Bakes-2/account/Dhash/images/new/".$cake_type."@/".$url;



$dbservername = "localhost";
$dbUsername = "root";
$dbPassword = "";
$dbName = "kaye-bakes";

$conn = mysqli_connect($dbservername,$dbUsername,$dbPassword,$dbName);


?>

<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Kaye Bakes - Customize</title>

    <link rel="stylesheet" type="text/css"  media="screen,projection" href="/Kaye-Bakes-2/account/css/new.css">
    <link rel="stylesheet" type="text/css" href="/Kaye-Bakes-2/account/css/bootstrap/bootstrap.min.css">
    <link rel="stylesheet" type="text/css" href="/Kaye-Bakes-2/account/css/font-awesome.min.css">
    <link rel="stylesheet" type="text/css" href="/Kaye-Bakes-2/account/css/list-groups.css">
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


    <header>
      <div class="navbar navbar-dark ground-dark shadow-sm">
      <div class="container">
          <a href="/Kaye-Bakes-2/index.html" class="navbar-brand d-flex align-items-center"><img src="/Kaye-Bakes-2/account/images/materialize-logo.png" alt="logo"></a> <span class="logo-text"></span>

          </button>
      </div>
      </div>
    </header>


    <main>

    <!--base dafault details-->
    <!--form-->

    <form method="post" action="/Kaye-Bakes-2/account/Dhash/custom-add-cart.php">
   
    <div class="container">
        <section class="py-5  container  w-50">
            <div class="row ">
                <div class="col">
                    <div class="card shadow-sm">
                    
                    <img class="img-thumbnail card-img-top"  src="<?php echo $basePath; ?>"  alt="uploaded image"/>

                    </div>
                </div>

                <div class="col ">
                <?php
                
                $sql = "SELECT * FROM products WHERE product_name = '".$baseName."' ";
                $result = mysqli_query($conn, $sql);
                $query_results = mysqli_fetch_array($result);
                
                if ($query_results > 0) {
                  ?>
                  <h1 class="bebas"><?php echo $query_results['product_name'];?></h1>
                  <p>Size: <?php 
                  $onel="";
                  $twol="";
                  $threel="";
                  $fouri="";
                  $sixi="";
                  $ninei="";
                  $twelvei="";
                  switch($query_results['cake_size']){
                    case '1layered':
                      echo "1 layer";
                      $onel="selected";
                      break;
                    case '2layered':
                      echo "2 layers";
                      $twol="selected";
                      break;
                    case '3layered':
                      echo "3 layers";
                      $threel="selected";
                      break;
                    case '4inches':
                      echo "4 inches";
                      $fouri="selected";
                      break;
                    case '6inches':
                      echo "6 inches";
                      $sixi="selected";
                      break;
                    case '9inches':
                      echo "9 inches";
                      $ninei="selected";
                      break;
                    case '12inches':
                      echo "12 inches";
                      $twelvei="selected";
                      break;      
                  }                  
                  ?></p>
                  <p>Flavor: <?php 
                  $chocolate="";
                  $red_velvet="";
                  $carrot="";
                  $mocha_chiffon="";
                  $vanilla_chiffon="";
                  $carrot_dog="";
                  switch($query_results['cake_flavor']){
                    case 'chocolate':
                      echo "Chocolate Flavor";
                      $chocolate="checked";
                      break;
                    case 'red_velvet':
                      echo "Red Velvet";
                      $red_velvet="checked";
                      break;
                    case 'carrot':
                      echo "Carrot Flavor";
                      $carrot="checked";
                      break;
                    case 'mocha_chiffon':
                      echo "Mocha Chiffon Flavor";
                      $mocha_chiffon="checked";
                      break;
                    case 'vanilla_chiffon':
                      echo "Vanilla Chiffon Flavor";
                      $vanilla_chiffon="checked";
                      break;
                    case 'carrot_dog':
                      echo "Carrot Flavor (Dog Cake)";
                      $carrot_dog="checked";
                      break;      
                  }
                  ?></p>
                  <p>Icing: <?php 
                  $buttercream ="";
                  $cream_cheese ="";
                  $meringue ="";
                  $chocolate_icing ="";
                  $fondant ="";
                  $potato ="";
                  switch($query_results['icing_flavor']){
                    case 'buttercream':
                      echo "Buttercream";
                      $buttercream="checked";
                      break;
                    case 'cream_cheese':
                      echo "Cream Cheese";
                      $cream_cheese="checked";
                      break;
                    case 'meringue':
                      echo "Meringue";
                      $meringue="checked";
                      break;
                    case 'chocolate_icing':
                      echo "Chocolate";
                      $chocolate_icing="checked";
                      break;
                    case 'fondant':
                      echo "Fondant";
                      $fondant="checked";
                      break;
                    case 'potato':
                      echo "Potato";
                      $potato="checked";
                      break;      
                  }
                  
                  ?></p>

                  <?php
                  }
                else {
                  ?>
                  !DATABASE ERROR!
                  <?php }

                  echo "<input hidden value ='".$query_results['product_id']."' id = 'id_old' name='id_old'>";
                  echo "<input hidden value ='".$query_results['product_name']."' id = 'name_old' name='name_old'>";
                  echo "<input hidden value ='".$query_results['product_desc']."' id = 'desc_old' name='desc_old'>";
                  echo "<input hidden value ='".$query_results['price']."' id = 'price_old' name='price_old'>";
                  echo "<input hidden value ='".$query_results['cake_type']."' id = 'cake_type_old' name='cake_type_old'>";
                  echo "<input hidden value ='".$query_results['cake_size']."' id = 'cake_size_old' name='cake_size_old'>";
                  echo "<input hidden value ='".$query_results['cake_flavor']."' id = 'cake_flavor_old' name='cake_flavor_old'>";
                  echo "<input hidden value ='".$query_results['icing_flavor']."' id = 'icing_flavor_old' name='icing_flavor_old'>";
                  echo "<input hidden value ='".$query_results['image_url']."' id = 'image_url_old' name='image_url_old'>";
                  echo "<input hidden value ='".$query_results['customized']."' id = 'customized_old' name='customized_old'>";
                    ?>
                </div>
            </div>
        </section>
    </div>


    
  <!--Customizations start-->

  <hr class="my-4">

    <div class="container ">
        <div class=" row  ">
            <h3 class="p-3 text-center display-5">Customize orders here</h3>  
        </div>
    </div>
    <hr class="my-4">


      <div class="row justify-content-md-center">
          <div class="col ">
            <div class="mx-auto" >
            <h4 class=" bebas mt-2 fs-1 text-center"> <?php 
            switch($_SESSION['cake_type']){
              case 'cake':
                echo "Regular Cake";
                break;
              case 'cupcakes':
                echo "Cupcakes";
                break;
              case 'number_cup':
                echo "Number Cupcakes";
                break;              
              default:
                echo $_SESSION['cake_type']." Cake";
                break;            } 
              ?></h4>
          </div>
          </div>
          
      <!-- select id=cake_size -->    
          <div class="col">
            <?php
            switch($_SESSION['cake_type']){
              case "layered":
                ?>
                <select name="cake_size" id="cake_size"  class="form-control  w-50 mt-3  w-50 mt-3" >
                  <option value="" disabled  hidden>Number of Layers</option>
                  <option value="2layered" <?php echo $twol;?> >Two-Layered Cake</option>
                  <option value="3layered" <?php echo $threel;?>>Three-Layered Cake</option>
                  </select>
           
                <?php
                break;
              case "cake":
                ?>
                <select name="cake_size" id="cake_size"  class="form-control  w-50 mt-3" >
                  <option value="" disabled  hidden>Cake Size</option>
                  <option value="6inches" <?php echo $sixi;?>>6 inches</option>
                  <option value="9inches" <?php echo $ninei;?> >9 inches</option>
                  <option value="12inches" <?php echo $twelvei;?>>12 inches</option>
                  </select>
                <?php
                break;
              case "dog":
                ?>
                <select name="cake_size" id="cake_size"  class="form-control  w-50 mt-3" >
                  <option value="" disabled hidden>Cake Size</option>
                  <option value="4inches" <?php echo $fouri;?>>4 inches</option>
                  <option value="6inches"<?php echo $sixi;?>>6 inches</option>
                  <option value="9inches"<?php echo $ninei;?>>9 inches</option>
                  </select>
                <?php
                break;
              case "cat":
                ?>
                <select name="cake_size" id="cake_size"  class="form-control  w-50 mt-3" >
                  <option value="" disabled >Cake Size</option>
                  <option value="4inches" selected >4 inches</option>
                  </select>
                <?php
                break;
              case "money":
                ?>
                <select name="cake_size" id="cake_size"  class="form-control  w-50 mt-3" >
                  <option value="" disabled selected hidden>Cake Size</option>
                  <option value="6inches" <?php echo $sixi;?>>6 inches</option>
                  <option value="9inches"<?php echo $ninei;?>>9 inches</option>
                  </select>
                <?php
                break;
              case "wedding":
                ?>
                <select name="cake_size" id="cake_size"  class="form-control  w-50 mt-3" >
                  <option value="" disabled selected hidden>Cake Size</option>
                  <option value="1layered" <?php echo $onel;?>>1 layer</option>
                  <option value="2layered" <?php echo $twol;?>>2 layers</option>
                  <option value="3layered" <?php echo $threel;?>>3 layers</option>
                  </select>
                <?php
                break;
              default:
                
                break;  }
              
            
                ?>
          </div>
      </div>
  </div>

  <hr class="my-4">
    <!--radio name="cake_flavor" -->
    <div class="container shadow-sm ">
        <div class="row  ">
            <h4 class="mt-2 ms-2 pt-5 display-6">Cake Flavor</h4>
        </div>
        <div class="row ">
            <div class="col ">
                <div class="list-group list-group-checkable d-grid gap-2 border-0 w-auto">
                    <input class="list-group-item-check pe-none" type="radio" name="cake_flavor" id="chocolate" value="chocolate" <?php echo $chocolate; ?> >
                    <label class="list-group-item rounded-3 py-3" for="chocolate">
                        Chocolate
                      <span class="d-block small opacity-50">With support text underneath to add more detail</span>
                    </label>
                  
                    <input class="list-group-item-check pe-none" type="radio" name="cake_flavor" id="red_velvet" value="red_velvet"  <?php echo $red_velvet; ?>>
                    <label class="list-group-item rounded-3 py-3" for="red_velvet">
                      Red Velvet
                      <span class="d-block small opacity-50">Some other text goes here</span>
     
                    </label>


                    <input class="list-group-item-check pe-none" type="radio" name="cake_flavor" id="carrot" value="carrot" <?php echo $carrot; ?>>
                    <label class="list-group-item rounded-3 py-3" for="carrot">
                      Carrot Cake
                      <span class="d-block small opacity-50">Some other text goes here</span>
     
                    </label>
                  </div>
            </div>
            <div class="col">
                <div class="list-group list-group-checkable d-grid gap-2 border-0 w-auto">
                  
                    <input class="list-group-item-check pe-none" type="radio" name="cake_flavor" id="mocha_chiffon" value="mocha_chiffon" <?php echo $mocha_chiffon; ?>>
                    <label class="list-group-item rounded-3 py-3" for="mocha_chiffon">
                      Mocha Chiffon
                      <span class="d-block small opacity-50">And we end with another snippet of text</span>
                    </label>
                  
                    <input class="list-group-item-check pe-none" type="radio" name="cake_flavor" id="vanilla_chiffon" value="vanilla_chiffon" <?php echo $vanilla_chiffon; ?>>
                    <label class="list-group-item rounded-3 py-3" for="vanilla_chiffon">
                      Vanilla Chiffon
                      <span class="d-block small opacity-50">details ...</span>
                    </label>

                    
                    <input class="list-group-item-check pe-none" type="radio" name="cake_flavor" id="carrot_dog" value="carrot_dog" <?php echo $carrot_dog; ?>>
                    <label class="list-group-item rounded-3 py-3" for="carrot_dog">
                      Carrot Cake (Dog Cake)
                      <span class="d-block small opacity-50">details ...</span>
                    </label>
                  </div>
            </div>

        </div>
    </div>


    <hr class="my-4">
    <!--radio name="icing"-->
    <div class="container shadow-sm">
        <div class="row  ">
            <h4 class="mt-2 ms-2 display-6">Icing</h4>
        </div>
        <div class="row ">
            <div class="col ">
                <div class="list-group list-group-checkable d-grid gap-2 border-0 w-auto">
                    <input class="list-group-item-check pe-none" type="radio" name="icing" id="buttercream" value="buttercream" <?php echo $buttercream;?> >
                    <label class="list-group-item rounded-3 py-3" for="buttercream">
                      Buttercream Frosting
                      <span class="d-block small opacity-50">With support text underneath to add more detail</span>
                    </label>
                  
                    <input class="list-group-item-check pe-none" type="radio" name="icing" id="cream_cheese" value="cream_cheese" <?php echo $cream_cheese;?>>
                    <label class="list-group-item rounded-3 py-3" for="cream_cheese">
                      Cream Cheese Frosting
                      <span class="d-block small opacity-50">Some other text goes here</span>
                    </label>

                    <input class="list-group-item-check pe-none" type="radio" name="icing" id="meringue" value="meringue" <?php echo $meringue;?>>
                    <label class="list-group-item rounded-3 py-3" for="meringue">
                      Meringue Icing
                      <span class="d-block small opacity-50">(applicable for cupcakes only)</span>
                    </label>      
                  </div>
            </div>
            <div class="col ">
                <div class="list-group list-group-checkable d-grid gap-2 border-0 w-auto">
                  
                    <input class="list-group-item-check pe-none" type="radio" name="icing" id="chocolate_icing" value="chocolate_icing" <?php echo $chocolate_icing;?>>
                    <label class="list-group-item rounded-3 py-3" for="chocolate_icing">
                      Chocolate Frosting
                      <span class="d-block small opacity-50">And we end with another snippet of text</span>
                    </label>
                  
                    <input class="list-group-item-check pe-none" type="radio" name="icing" id="potato" value="potato" <?php echo $potato;?> >
                    <label class="list-group-item rounded-3 py-3" for="potato">
                      Potato Icing (for Dog Cake)
                      <span class="d-block small opacity-50">This option is disabled</span>
                    </label>
                 
                    <input class="list-group-item-check pe-none" type="radio" name="icing" id="fondant" value="fondant" <?php echo $fondant;?> >
                    <label class="list-group-item rounded-3 py-3" for="fondant">
                      Fondant Icing
                      <span class="d-block small opacity-50">text text</span>
                    </label>
                 
                  </div>
            </div>

        </div>
    </div>

    <hr class="my-4">
    <!--quantity number id="quantity" -->

    <div class="container">
        <div class="row">
            <div class="col">
              <h4 class=" text-end me-5 mt-1 display-6">Quantity</h4>
            </div>
            <div class="col col-md-3">
                <div class="">
                    <input class="text-center w-50" type="number" name="quantity" id="quantity" min="1" value=1>
            </div>
        </div>
    </div>
    <div class="container ">
      <div class="row justify-content-center ">
        <div class="col col-md-4 ">
        <button type="submit" class="btn btn-lg btn-dark py-10" value="post">Add to Cart</button>
      </div>
      </form>
      </div>

    </div>
               
    

 

    
</main>

<svg xmlns="http://www.w3.org/2000/svg" style="display: none;">
  <symbol id="bootstrap" viewBox="0 0 118 94">
    <title>Bootstrap</title>
    <path fill-rule="evenodd" clip-rule="evenodd" d="M24.509 0c-6.733 0-11.715 5.893-11.492 12.284.214 6.14-.064 14.092-2.066 20.577C8.943 39.365 5.547 43.485 0 44.014v5.972c5.547.529 8.943 4.649 10.951 11.153 2.002 6.485 2.28 14.437 2.066 20.577C12.794 88.106 17.776 94 24.51 94H93.5c6.733 0 11.714-5.893 11.491-12.284-.214-6.14.064-14.092 2.066-20.577 2.009-6.504 5.396-10.624 10.943-11.153v-5.972c-5.547-.529-8.934-4.649-10.943-11.153-2.002-6.484-2.28-14.437-2.066-20.577C105.214 5.894 100.233 0 93.5 0H24.508zM80 57.863C80 66.663 73.436 72 62.543 72H44a2 2 0 01-2-2V24a2 2 0 012-2h18.437c9.083 0 15.044 4.92 15.044 12.474 0 5.302-4.01 10.049-9.119 10.88v.277C75.317 46.394 80 51.21 80 57.863zM60.521 28.34H49.948v14.934h8.905c6.884 0 10.68-2.772 10.68-7.727 0-4.643-3.264-7.207-9.012-7.207zM49.948 49.2v16.458H60.91c7.167 0 10.964-2.876 10.964-8.281 0-5.406-3.903-8.178-11.425-8.178H49.948z"></path>
  </symbol>
  <symbol id="facebook" viewBox="0 0 16 16">
    <path d="M16 8.049c0-4.446-3.582-8.05-8-8.05C3.58 0-.002 3.603-.002 8.05c0 4.017 2.926 7.347 6.75 7.951v-5.625h-2.03V8.05H6.75V6.275c0-2.017 1.195-3.131 3.022-3.131.876 0 1.791.157 1.791.157v1.98h-1.009c-.993 0-1.303.621-1.303 1.258v1.51h2.218l-.354 2.326H9.25V16c3.824-.604 6.75-3.934 6.75-7.951z"/>
  </symbol>
  <symbol id="instagram" viewBox="0 0 16 16">
      <path d="M8 0C5.829 0 5.556.01 4.703.048 3.85.088 3.269.222 2.76.42a3.917 3.917 0 0 0-1.417.923A3.927 3.927 0 0 0 .42 2.76C.222 3.268.087 3.85.048 4.7.01 5.555 0 5.827 0 8.001c0 2.172.01 2.444.048 3.297.04.852.174 1.433.372 1.942.205.526.478.972.923 1.417.444.445.89.719 1.416.923.51.198 1.09.333 1.942.372C5.555 15.99 5.827 16 8 16s2.444-.01 3.298-.048c.851-.04 1.434-.174 1.943-.372a3.916 3.916 0 0 0 1.416-.923c.445-.445.718-.891.923-1.417.197-.509.332-1.09.372-1.942C15.99 10.445 16 10.173 16 8s-.01-2.445-.048-3.299c-.04-.851-.175-1.433-.372-1.941a3.926 3.926 0 0 0-.923-1.417A3.911 3.911 0 0 0 13.24.42c-.51-.198-1.092-.333-1.943-.372C10.443.01 10.172 0 7.998 0h.003zm-.717 1.442h.718c2.136 0 2.389.007 3.232.046.78.035 1.204.166 1.486.275.373.145.64.319.92.599.28.28.453.546.598.92.11.281.24.705.275 1.485.039.843.047 1.096.047 3.231s-.008 2.389-.047 3.232c-.035.78-.166 1.203-.275 1.485a2.47 2.47 0 0 1-.599.919c-.28.28-.546.453-.92.598-.28.11-.704.24-1.485.276-.843.038-1.096.047-3.232.047s-2.39-.009-3.233-.047c-.78-.036-1.203-.166-1.485-.276a2.478 2.478 0 0 1-.92-.598 2.48 2.48 0 0 1-.6-.92c-.109-.281-.24-.705-.275-1.485-.038-.843-.046-1.096-.046-3.233 0-2.136.008-2.388.046-3.231.036-.78.166-1.204.276-1.486.145-.373.319-.64.599-.92.28-.28.546-.453.92-.598.282-.11.705-.24 1.485-.276.738-.034 1.024-.044 2.515-.045v.002zm4.988 1.328a.96.96 0 1 0 0 1.92.96.96 0 0 0 0-1.92zm-4.27 1.122a4.109 4.109 0 1 0 0 8.217 4.109 4.109 0 0 0 0-8.217zm0 1.441a2.667 2.667 0 1 1 0 5.334 2.667 2.667 0 0 1 0-5.334z"/>
  </symbol>
  <symbol id="twitter" viewBox="0 0 16 16">
    <path d="M5.026 15c6.038 0 9.341-5.003 9.341-9.334 0-.14 0-.282-.006-.422A6.685 6.685 0 0 0 16 3.542a6.658 6.658 0 0 1-1.889.518 3.301 3.301 0 0 0 1.447-1.817 6.533 6.533 0 0 1-2.087.793A3.286 3.286 0 0 0 7.875 6.03a9.325 9.325 0 0 1-6.767-3.429 3.289 3.289 0 0 0 1.018 4.382A3.323 3.323 0 0 1 .64 6.575v.045a3.288 3.288 0 0 0 2.632 3.218 3.203 3.203 0 0 1-.865.115 3.23 3.23 0 0 1-.614-.057 3.283 3.283 0 0 0 3.067 2.277A6.588 6.588 0 0 1 .78 13.58a6.32 6.32 0 0 1-.78-.045A9.344 9.344 0 0 0 5.026 15z"/>
  </symbol>
</svg>

  <div class="container">
    <footer class="d-flex flex-wrap justify-content-between align-items-center py-3 my-4 border-top">
      <div class="col-md-4 d-flex align-items-center">
        <span class="mb-3 mb-md-0 text-muted">&copy; 2022 Kaye Bakes</span>
      </div>

      <ul class="nav col-md-4 justify-content-end list-unstyled d-flex">
        <li class="ms-3"><a class="text-muted" href="#"><svg class="bi" width="24" height="24"><use xlink:href="#twitter"/></svg></a></li>
        <li class="ms-3"><a class="text-muted" href="#"><svg class="bi" width="24" height="24"><use xlink:href="#instagram"/></svg></a></li>
        <li class="ms-3"><a class="text-muted" href="#"><svg class="bi" width="24" height="24"><use xlink:href="#facebook"/></svg></a></li>
      </ul>
    </footer>

 <!--bootstrap-->
 <script type="text/javascript" src="/Kaye-Bakes-2/account/js/bootstrap/bootstrap.bundle.min.js"></script>  

<!-- jQuery Library -->
<script type="text/javascript" src="/Kaye-Bakes-2/account/js/jquery-3.4.1.min.js"></script>

<!--bootstrap-->
<script type="text/javascript" src="/Kaye-Bakes-2/account/js/bootstrap/bootstrap-datepicker.min.js"></script>

<!--custom-->
<script type="text/javascript" src="/Kaye-Bakes-2/account/js/own-script.js"></script>

<script>

  $(window).on('load', function () {
    $('#loading').fadeOut("slow");
    }) 
  </script>
  </body>
</html>
