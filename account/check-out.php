<?php
session_start();
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
    <meta name="description" content="">
    <meta name="author" content="Mark Otto, Jacob Thornton, and Bootstrap contributors">
    <meta name="generator" content="Hugo 0.98.0">
    <title>Checkout example · Bootstrap v5.2</title>

    <link rel="canonical" href="https://getbootstrap.com/docs/5.2/examples/checkout/">
    <link rel="stylesheet" type="text/css" href="css/bootstrap/bootstrap.min.css">
    <link rel="stylesheet" type="text/css" href="css/font-awesome.min.css">
    <link rel="stylesheet" type="text/css" href="css/checkout.min.css">


    <!-- Custom styles for this template -->
    <link href="form-validation.css" rel="stylesheet">
</head>

<body class="">

    <div class="container">
        <main>
            <div class="py-5 text-center">
                <img class="d-block mx-auto mb-4" src="" alt="" width="" height="">
                <h2>Checkout form</h2>
                <p class="lead"></p>
            </div>

            <div class="row g-5">
      <div class="col-md-5 col-lg-4 order-md-last">
        <h4 class="d-flex justify-content-between align-items-center mb-3">
          <span class="text-primary">Your cart</span>
          <span class="badge bg-primary rounded-pill">3</span>
        </h4>
        <?php
          $total = 0;
          $cart_items = array('');
          $output = "";
          $output .= "
          <div class=row-md-4>
          <table class='table table-bordered table-stripped'>
                        <tr>
                            <th>Image</th>
                            <th>Name</th>
                            <th>Quantity</th>
                            <th>Price</th>
                        </tr>
                    ";

        if (!empty($_SESSION['cart'])) {
          foreach ($_SESSION['cart'] as $key => $value) {
              $output .= "
                  <tr>
                      <td>
                        <img src='Dhash/images/new/menu@/".$value['image_url']."' style='height: 100px'>
                      </td>
                      <td>".$value['product_name']."</td>
                      <td>".$value['quantity']."</td>
                      <td>Php  ".number_format($value['price'] * $value['quantity'])."</td>
                  </tr>
              ";
              $total = $total + $value['quantity'] * $value['price'];
              $all_cart_items = $value['product_id'];
          }
          $output .= "
                        <tr>
                          <td colspan='3'>Total Price</td>
                          <td>".number_format($total)."</td>
                        </tr>
                        </table>
                        </div>
                        ";
                        
        }
        echo $output
        ?>
        </div>
                    <div class="col-md-7 col-lg-8">
                        <div class="form-container">
                            <h4 class="mb-3">Billing address</h4>
                            <form class="needs-validation" novalidate>
                                <div class="row g-3">
                                    <div class="col-sm-12">
                                        <label for="firstName" class="form-label">Full Name</label>
                                        <input type="text" class="form-control" placeholder="" value="full name"
                                            required>
                                        <div class="invalid-feedback">
                                            Valid first name is required.
                                        </div>
                                    </div>

                                  <!--
                                    <div class="col-12">
                                        <label for="username" class="form-label">Username</label>
                                        <div class="input-group has-validation">
                                            <span class="input-group-text">@</span>
                                            <input type="text" class="form-control" id="username" placeholder="Username"
                                                required>
                                            <div class="invalid-feedback">
                                                Your username is required.
                                            </div>
                                        </div>
                                    </div>
                                  --->

                                    <div class="col-12">
                                        <label for="email" class="form-label">Email <span
                                                class="text-muted"></span></label>
                                        <input type="email" class="form-control" id="email"
                                            placeholder="you@example.com" value="email">
                                        <div class="invalid-feedback">
                                            Please enter a valid email address for shipping updates.
                                        </div>
                                    </div>

                                    <div class="col-12">
                                        <label for="address" class="form-label">Address</label>
                                        <input type="text" class="form-control" id="address" placeholder="1234 Main St"
                                            required value="address">
                                        <div class="invalid-feedback">
                                            Please enter your shipping address.
                                        </div>
                                    </div>

                                    <div class="col-12">
                                        <label for="address2" class="form-label">Contact Info. <span
                                                class="text-muted"></span></label>
                                        <input type="text" class="form-control" id="address2"
                                            placeholder="Cellphone or Telephone" value="contact">
                                    </div>

                                    <div class="col-md-5">
                                        <label for="country" class="form-label">Region</label>
                                        <select class="form-select" id="country" required>
                                            <option value="">Choose...</option>
                                            <option>Metro Manila</option>
                                            <option>Cavite</option>
                                        </select>
                                        <div class="invalid-feedback">
                                            Please select a valid Region.
                                        </div>
                                    </div>

                                    <div class="col-md-4">
                                        <label for="state" class="form-label">City</label>
                                        <select class="form-select" id="state" required>
                                            <option value="">Choose...</option>
                                            <option>Las Piñas</option>
                                            <option>Muntinlupa</option>
                                            <option>Parañaque</option>
                                            <option>Taguig</option>
                                            <option>Pasay</option>
                                            <option>Cavite</option>
                                            <option>Kawit</option>
                                            <option>Novelata</option>
                                            <option>Rosario</option>
                                            <option>Bacoor</option>
                                            <option>Imus</option>
                                            <option>Dasmariñas</option>
                                            <option>Others</option>

                                        </select>
                                        <div class="invalid-feedback">
                                            Please provide a valid City.
                                        </div>
                                    </div>

                                    <div class="col-md-3">
                                        <label for="zip" class="form-label">Zip</label>
                                        <input type="text" class="form-control" id="zip" placeholder="" required>
                                        <div class="invalid-feedback">
                                            Zip code required.
                                        </div>
                                    </div>
                                </div>

                                <hr class="my-4">

                                <div class="form-check">
                                    <input type="checkbox" class="form-check-input" id="same-address">
                                    <label class="form-check-label" for="same-address">Shipping address is the same as
                                        my billing address</label>
                                </div>

                                <hr class="my-4">

                                <h4 class="mb-3">Payment</h4>

                                <div class="my-3">
                                    <div class="form-check">
                                        <input id="credit" name="paymentMethod" type="radio" class="form-check-input"
                                            checked required>
                                        <label class="form-check-label" for="credit">Gcash</label>
                                    </div>
                                    <div class="form-check">
                                        <input id="debit" name="paymentMethod" type="radio" class="form-check-input"
                                            required>
                                        <label class="form-check-label" for="debit">Debit Card</label>
                                    </div>
                                    <div class="form-check">
                                        <input id="paypal" name="paymentMethod" type="radio" class="form-check-input"
                                            required>
                                        <label class="form-check-label" for="paypal">COD</label>
                                    </div>
                                </div>

                                <div class="row gy-3">
                                    <div class="col-md-6">
                                        <label for="cc-name" class="form-label">Name on card</label>
                                        <input type="text" class="form-control" id="cc-name" placeholder="" required>
                                        <small class="text-muted">Full name as displayed on card</small>
                                        <div class="invalid-feedback">
                                            Name on card is required
                                        </div>
                                    </div>

                                    <div class="col-md-6">
                                        <label for="cc-number" class="form-label">Credit card number</label>
                                        <input type="text" class="form-control" id="cc-number" placeholder="" required>
                                        <div class="invalid-feedback">
                                            Credit card number is required
                                        </div>
                                    </div>

                                    <div class="col-md-3">
                                        <label for="cc-expiration" class="form-label">Expiration</label>
                                        <input type="text" class="form-control" id="cc-expiration" placeholder=""
                                            required>
                                        <div class="invalid-feedback">
                                            Expiration date required
                                        </div>
                                    </div>

                                    <div class="col-md-3">
                                        <label for="cc-cvv" class="form-label">CVV</label>
                                        <input type="text" class="form-control" id="cc-cvv" placeholder="" required>
                                        <div class="invalid-feedback">
                                            Security code required
                                        </div>
                                    </div>
                                </div>

                                <hr class="my-4">

                                <button class="w-100 btn btn-primary btn-lg" type="submit" name="check-out">Continue to checkout</button>
                            </form>
                        </div>
                    </div>
                </div>
            </div>
        </main>


        <footer class="my-5 pt-5 text-muted text-center text-small">
            <p class="mb-1">&copy; 2018–2022 Kaye Bakes</p>
            <ul class="list-inline">
                <li class="list-inline-item"><a href="#">Privacy</a></li>
                <li class="list-inline-item"><a href="#">Terms</a></li>
                <li class="list-inline-item"><a href="#">Support</a></li>
            </ul>
        </footer>
    </div>


    <script src="../assets/dist/js/bootstrap.bundle.min.js"></script>

    <script src="form-validation.js"></script>
</body>

</html>