<?php
session_start();
?>



<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <title>Kaye Bakes - Checkout</title>

    <link rel="stylesheet" type="text/css"  media="screen,projection" href="/Kaye-Bakes-2/account/css/new.css">
    <link rel="stylesheet" type="text/css" href="/Kaye-Bakes-2/account/css/bootstrap/bootstrap.min.css">
    <link rel="stylesheet" type="text/css" href="/Kaye-Bakes-2/account/css/bootstrap/form-validation.css">
    <link rel="stylesheet" type="text/css" href="/Kaye-Bakes-2/account/css/list-groups.css">
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


  <body class="bg-light">
    <!--Loader-->
      <div class="loader" id="loading"> 
        <div class="cssload-container" id="loading-image">
          <div class="cssload-circle"></div>
          <div class="cssload-circle"></div>
        </div>
      </div>


 
<div class="container">
  <main>

    <!--form-->
    <form class="needs-validation" action="check-data.php" method="post" novalidate>

    <div class="py-5 text-center">
<!--      <img class="d-block mx-auto mb-4" src="../assets/brand/bootstrap-logo.svg" alt="" width="72" height="57"> -->
      <h2>Checkout form</h2>
      <p class="lead">Details details</p>
    </div>

    
    <div class="row g-5">
      <!--mini cart--> 
        <div class="col-md-5 col-lg-4 order-md-last">
          <h4 class="d-flex justify-content-between align-items-center mb-3">
            <span class="text-primary">Your cart</span>
            <span class="badge bg-primary rounded-pill"><?php echo count($_SESSION['cart']); ?></span>
          </h4>
          <ul class="list-group mb-3">
          <?php
          $total=0;
          if (!empty($_SESSION['cart'])) {
            foreach ($_SESSION['cart'] as $key => $value) {
              ?>
              
              <li class="list-group-item d-flex justify-content-between lh-sm">
                <div>
                  <h6 class="my-0"><?php echo $value['product_name'].'('.$value['quantity'].')'; ?></h6>
                  <small class="text-muted"><?php echo $value['product_desc']; ?></small>
                </div>
              <span class="text-muted">Php <?php echo number_format($value['price'] * $value['quantity']);?></span>
              </li>

              <?php
              $total = $total + $value['quantity'] * $value['price'];
            }
          }
          else{
            echo "Error in retreiving Cart Items.";
          }
          
          ?>

          
            <li class="list-group-item d-flex justify-content-between">
              <span>Total </span>
              <strong>Php <?php echo $total; ?></strong>
            </li>
          </ul>

      <!-- textarea id="dedications"-->
          <hr class="my-4">
          <hr class="my-4">

          <div class="container">
            <label for="dedications" class="form-label">Dedications (optional)</label>
            <textarea class="form-control" id="dedications" name="dedications" rows="4"  placeholder="ex. Happy 35th Birthday Mom!"></textarea>
          </div>
          
          <hr class="my-4">

      <!-- textarea id="requests_details"-->
          <div class="container">
            <label for="requests_details" class="form-label">Other Requests (optional)</label>
            <textarea class="form-control" id="requests_details" name="requests_details" rows="4" placeholder=" ex. Topper- Peppa Pig"></textarea>
          </div>






        </div>

        


      
      
      
      <div class="col-md-7 col-lg-8">
        <h4 class="mb-3">Basic Information</h4>
        
      
          <div class="row g-3">
      <!-- Name id= firstName, lastName-->
            <div class="col-sm-6">
              <label for="firstName" class="form-label">First name</label>
              <input type="text" class="form-control" id="firstName" name="firstName" placeholder=""  required >
              <div class="invalid-feedback">
                Valid first name is required.
              </div>
            </div>

            <div class="col-sm-6">
              <label for="lastName" class="form-label">Last name</label>
              <input type="text" class="form-control" id="lastName" name="lastName" placeholder="" value="" required >
              <div class="invalid-feedback">
                Valid last name is required.
              </div>
            </div>

      <!-- contact id=mobile-->
            
            <div class="col-12">
              <label for="mobile" class="form-label">Contact Number</label>
              <div class="input-group has-validation">
                <span class="input-group-text">+63</span>
                <input type="number" class="form-control" id="mobile" name="mobile" placeholder="9XX XXX XXXX " required min=1000000000 max=9999999999>
              <div class="invalid-feedback">
                  Valid contact number is required.
                </div>
              </div>
            </div>

      <!-- id= email-->
            

            <div class="col-12">
              <label for="email" class="form-label">Email <span class="text-muted"></span></label>
              <input type="email" class="form-control" id="email" name="email" placeholder="you@example.com" required>
              <div class="invalid-feedback">
                Please enter a valid email address for shipping updates.
              </div>
            </div>


            <hr class="my-4">

              <h4 class="mb-3">Address</h4>
           
      <!-- id="province"  id="city"-->
            
            <div class="col-md-6">
              <label for="province" class="form-label">Province</label>
              <select class="form-select" id="province" name="province" required>
                <option value="">Choose...</option>
                <option value="NCR">Metro Manila</option>
                <option value="Cavite">Cavite</option>
              </select>
              <div class="invalid-feedback">
                Please select a valid Province.
              </div>
            </div>

            <div class="col-md-6">
              <label for="city" class="form-label">City</label>
              <select class="form-select" id="city" name="city" required>
                <option value="">Choose...</option>
                <option value="Cavite City">Cavite City</option>
                <option value="Kawit">Kawit</option>
                <option value="Rosario">Rosario</option>
              </select>
              <div class="invalid-feedback">
                Please provide a valid City.
              </div>
            </div>

          
  
      <!-- id="address_specific"-->

            <div class="col-12">
              <label for="address_specific" class="form-label">House # and Street</label>
              <input type="text" class="form-control" id="address_specific" name="address_specific" placeholder="798, Mary St. Liverpool Subdivision" required>
              <div class="invalid-feedback">
                Please enter your shipping address.
              </div>
            </div>
      

            <hr class="my-4">

            <h4 class="mb-3">Delivery</h4> 
            <!-- time id=time_delivery-->
                <div class="container">
                  <div class="col-4">
                    <label for="time_delivery" class="form-label">Estimated time of delivery</label>
                    <input type="time" class="form-control" id="time_delivery" name="time_delivery" required >
                      <div class="invalid-feedback">
                        Valid time of delivery is required.
                      </div>
                  </div>
                  <div class="col-8"></div>
                </div>

            <hr class="my-4">



          <h4 class="mb-3">Shipping Mode</h4>  
          

          <div class="container">
  
            <div class="row ">

                <div class="col col-md-4">
                    <div class="list-group list-group-checkable d-grid gap-2 border-0 w-auto">
                        <input class="form-control list-group-item-check pe-none" type="radio" name="shipping_mode" id="motor" value="motor" onclick="show();" required >
                        <label class="form-check-label list-group-item rounded-3 py-3" for="motor">
                          Motorcycle
                          <span class="d-block small opacity-50">Lalamove, Grab or etc. Cheaper but prone to damage</span>
                        </label>
                        <div class="invalid-feedback">
                          Please enter your shipping mode.
                        </div>
                      </div>
                  
                </div>

                <div class="col col-md-4">
                    <div class="list-group list-group-checkable d-grid gap-2 border-0 w-auto">
                      
                        <input class="list-group-item-check pe-none" type="radio" name="shipping_mode" id="car" value="car" onclick="show();" >
                        <label class="list-group-item rounded-3 py-3" for="car">
                          Car
                          <span class="d-block small opacity-50">Lalamove, Grab or etc. A bit costly but safer and suggested.</span>
                        </label>

                      </div>
                </div>

                <div class="col col-md-4">
                  <div class="list-group list-group-checkable d-grid gap-2 border-0 w-auto">
                    
                      <input class="list-group-item-check pe-none" type="radio" name="shipping_mode" id="pick-up" value="pick-up" onclick="hide();">
                      <label class="list-group-item rounded-3 py-3" for="pick-up">
                        Pick- up
                        <span class="d-block small opacity-50">Pick- up at Kaye Bakes' location</span>
                      </label>

                    </div>
              </div>
    
            </div>
        </div>


         

            <div id="delivery" style='display:none'>
              <hr class="my-4">

            <h4 class="mb-3">Delivery Booking Options</h4> 

            <div class="container">
  
              <div class="row ">
  
                  <div class="col col-md-6">
                      <div class="list-group list-group-checkable d-grid gap-2 border-0 w-auto">
                          <input class="list-group-item-check pe-none" type="radio" name="delivery_options" id="own" value="own" onclick="hide2();" >
                          <label class="list-group-item rounded-3 py-3" for="own">
                            Own Delivery
                            <span class="d-block small opacity-50">I will book and pay my delivery through Lalamove, Grab, etc.</span>
                          </label>
                    </div>
                  </div>
  
                  <div class="col col-md-6">
                      <div class="list-group list-group-checkable d-grid gap-2 border-0 w-auto">
                        
                          <input class="list-group-item-check pe-none" type="radio" name="delivery_options" id="shop" value="shop" onclick="show2();" >
                          <label class="list-group-item rounded-3 py-3" for="shop">
                            Shop's Delivery
                            <span class="d-block small opacity-50">Kaye Bakes will book my delivery through Lalamove, Grab, etc.</span>
                          </label>
  
                        </div>
                  </div>
  

              </div>
            </div>
          </div>



          


            <div id="shipFee" style='display:none'>
              <hr class="my-4">
            <h4 class="mb-3">Shipping Fee Options</h4> 

            <div class="container">
  
              <div class="row ">
  
                  <div class="col col-md-6">
                      <div class="list-group list-group-checkable d-grid gap-2 border-0 w-auto">
                          <input class="list-group-item-check pe-none" type="radio" name="shipping_fee" id="ownPay" value="ownPay"  >
                          <label class="list-group-item rounded-3 py-3" for="ownPay">
                            Own Payment
                            <span class="d-block small opacity-50">I will directly pay the rider for the shipping fee when my order arrives</span>
                          </label>
                    </div>
                  </div>
  
                  <div class="col col-md-6">
                      <div class="list-group list-group-checkable d-grid gap-2 border-0 w-auto">
                        
                          <input class="list-group-item-check pe-none" type="radio" name="shipping_fee" id="included" value="included"  >
                          <label class="list-group-item rounded-3 py-3" for="included">
                            Included
                            <span class="d-block small opacity-50">Please include the delivery fee in my total payment.</span>
                          </label>
  
                        </div>
                  </div>
  

              </div>
            </div>
          </div>
          <hr class="my-4">



          <h4 class="mb-3">Payment</h4>

          <div class="my-3">
            <div class="form-check">
              <input id="gcash" value="gcash" name="payment" type="radio" class="form-check-input" required>
              <label class="form-check-label" for="credit">GCash</label>
            </div>
            <div class="form-check">
              <input id="bank" value="bank" name="payment" type="radio" class="form-check-input" required>
              <label class="form-check-label" for="debit">Bank Transfer</label>
            </div>



          <hr class="my-4">

          <button class="w-100 btn btn-primary btn-lg" type="submit">Continue to checkout</button>
        </form>
      </div>
    </div>
  </main>

  <footer class="my-5 pt-5 text-muted text-center text-small">
    <p class="mb-1">&copy; 2017–2022 Company Name</p>
    <ul class="list-inline">
      <li class="list-inline-item"><a href="#">Privacy</a></li>
      <li class="list-inline-item"><a href="#">Terms</a></li>
      <li class="list-inline-item"><a href="#">Support</a></li>
    </ul>
  </footer>
</div>
<script>

function show(){
  document.getElementById('delivery').style.display ='block';
}
function hide(){
  document.getElementById('delivery').style.display ='none';
}

function show2(){
  document.getElementById('shipFee').style.display ='block';
}
function hide2(){
  document.getElementById('shipFee').style.display ='none';
}
</script>

 <!--bootstrap-->
 <script type="text/javascript" src="/Kaye-Bakes-2/account/js/bootstrap/bootstrap.bundle.min.js"></script>  
 <script type="text/javascript" src="/Kaye-Bakes-2/account/js/bootstrap/form-validation.js"></script>  

<!-- jQuery Library -->
<script type="text/javascript" src="/Kaye-Bakes-2/account/js/jquery-3.4.1.min.js"></script>


<!--custom-->
<script type="text/javascript" src="js/own-script.js"></script>
<script>
$(window).on('load', function () {
  $('#loading').fadeOut("slow");
  }) 
</script>
  </body>
</html>
