<?php

session_start();

$dbservername = "localhost";
$dbUsername = "root";
$dbPassword = "";
$dbName = "kaye-bakes";
$conn = mysqli_connect($dbservername,$dbUsername,$dbPassword,$dbName);

if (isset($_POST['check-out'])) {

    if (isset($_SESSION['cart-2'])) {
        

    } else {
        $session_array = array (
            'product_id'=> $_GET['product_id'],
            "product_name" => $_POST['product_name'],
            "product_desc" => $_POST['product_desc'],
            "total-price" => $total

        );
        $_SESSION['cart-2'][] = $session_array;
    }
}
header("location: ../check-out.php");
?>