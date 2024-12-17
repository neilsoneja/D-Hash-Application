<?php

session_start();

$dbservername = "localhost";
$dbUsername = "root";
$dbPassword = "";
$dbName = "kaye-bakes";
$conn = mysqli_connect($dbservername,$dbUsername,$dbPassword,$dbName);

  $msg = "";
  

  // If upload button is clicked ...
  if (isset($_POST['check-out'])) {
  	// Get image name
    $name = $_POST['name'];
    $email = $_POST['email'];
    $address = $_POST['address'];
    $contact = $_POST['contact'];
    $payment = $_POST['paymentMethod'];
    $total;
  	// image file directory
  	$sql = "INSERT INTO orders(order_id, customer_id, address, date, payment, order_type, datetime_delivery,
                               delivery_options, shipping_fee, total, dedications, request_details, deleted, status) 
      
      VALUES('', '', '', '$address', '$payment', '', '', '', '', '$total', '', '', '', '')";
  	// execute query
  	mysqli_query($db, $sql);
  }
  ?>