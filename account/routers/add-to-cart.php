<?php

session_start();

$dbservername = "localhost";
$dbUsername = "root";
$dbPassword = "";
$dbName = "kaye-bakes";
$conn = mysqli_connect($dbservername,$dbUsername,$dbPassword,$dbName);

if (isset($_POST['add-to-cart'])) {


        

        $session_array = array (
            "product_id"=> $_POST['product_id'],
            "image_url" => $_POST['image_url'],
            "product_name" => $_POST['product_name'],
            "product_desc" => $_POST['product_desc'],
            "price" => $_POST['price'],
            "quantity" => $_POST['quantity']
        );
        $_SESSION['cart'][] = $session_array;
    
}
header("location: ../cart-new.php");
?>
