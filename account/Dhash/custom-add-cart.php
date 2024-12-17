<?php
session_start();




$name=$_POST['name_old'];
$id=$_POST['id_old'] + (100*$_SESSION['uploadNumber']);
$desc=$_POST['desc_old'];
$price=$_POST['price_old'];
$cake_type=$_POST['cake_type_old'];
$cake_size=$_POST['cake_size_old'];
$cake_flavor=$_POST['cake_flavor_old'];
$icing_flavor=$_POST['icing_flavor_old'];
$customized=$_POST['customized_old'];



if (isset($_POST['cake_size'])){
    $cake_size=$_POST['cake_size'];
}

if (isset($_POST['cake_flavor'])){
    $cake_flavor=$_POST['cake_flavor'];
}
if (isset($_POST['icing'])){
    $icing_flavor=$_POST['icing'];
}
if (isset($_POST['cake_size'])){
    $cake_size=$_POST['cake_size'];
}



$order_array = array(
    'product_id'  => $id,
    'product_name' => $name,
    'product_desc' => $desc,
    'price' => $price,
    'cake_type' => $cake_type,
    'cake_size' => $cake_size,
    'cake_flavor' => $cake_flavor,
    'icing_flavor' => $icing_flavor,
    'customized' => $customized,
    'quantity' => $_POST['quantity'],
    'image_url' => $_SESSION['queryPath']);
$_SESSION['cart'][] = $order_array;
header("location: /Kaye-Bakes-2/account/cart-new.php");
?>