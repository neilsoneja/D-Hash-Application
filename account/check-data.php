<?php
session_start();
date_default_timezone_set("Asia/Tokyo");


$dbservername = "localhost";
$dbUsername = "root";
$dbPassword = "";
$dbName = "kaye-bakes";
$conn = mysqli_connect($dbservername,$dbUsername,$dbPassword,$dbName);

//customer table
$lastName=strtolower($_POST['lastName']);
$firstName=strtolower($_POST['firstName']);
$address_main=strtolower($_POST['province'].', '.$_POST['city']);
$address_specific=strtolower($_POST['address_specific']);
$email=$_POST['email'];
$mobile=$_POST['mobile'];



$sql = "SET FOREIGN_KEY_CHECKS=0;";
$result = mysqli_query($conn, $sql);


//check if customer exists
$sql = "SELECT * FROM customers WHERE address_main = '".$address_main."'  AND lastName = '".$lastName."' AND email='".$email."'";
$result = mysqli_query($conn, $sql);
$query_results = mysqli_fetch_assoc($result);







if($query_results == 0){

    $sql = "INSERT INTO customers ( lastName, firstName, address_main, address_specific, email, mobile)  
    VALUES ('".$lastName."', '".$firstName."', '".$address_main."', '".$address_specific."', '".$email."', '".$mobile."')";
 
    if(mysqli_query($conn, $sql)){
        //echo "Customer data import success";
        $sql = "SELECT customer_id FROM customers WHERE address_main = '".$address_main."'  AND lastName = '".$lastName."' AND email='".$email."'";
        $result = mysqli_query($conn, $sql);
        $query_results = mysqli_fetch_array($result);
        $customer_id= $query_results['customer_id'];
    }
    else{
        //echo "Customer Data Import Error";
    }
}
else{
    $customer_id= $query_results['customer_id'];

}

//order type
$menu=False; $custom=False;

$total=0;

if (!empty($_SESSION['cart'])) {
    foreach ($_SESSION['cart'] as $key => $value) {
        if ($value['customized']==1){
            $custom=True;
        }
        if ($value['customized']==0){
            $menu=True;
        }
        $total = $total + $value['quantity'] * $value['price'];
    }
    
    if ($custom and $menu ){
        $order_type='hybrid';
        $reference_in= "H";
    }
    elseif ($custom){
        $order_type='custom';
        $reference_in= "C";
    }
    elseif ($menu){
        $order_type='menu';
        $reference_in= "M";
    }
    else{
        //echo "order type error.";
    }
}
else{
    //echo"Shopping Cart Empty.";

}

//orders table

if (!isset($_POST['delivery_options'])){
    $delivery_options="";
}
else{
    $delivery_options=$_POST['delivery_options'];
}

if (!isset($_POST['shipping_fee'])){
    $shipping_fee="";
}
else{
    $shipping_fee=$_POST['shipping_fee'];
}

if (!isset($_POST['dedications'])){
    $dedications="";
}
else{
    $dedications=$_POST['dedications'];
}

if (!isset($_POST['requests_details'])){
    $requests_details="";
}
else{
    $requests_details=$_POST['requests_details'];
}

$payment=$_POST['payment'];
$shipping_mode= $_POST['shipping_mode'];
$order_date = date('y-m-d');
$time_delivery=$_POST['time_delivery'];
$date_delivery=$_SESSION["date_delivery"];
$order_status='pending';


$sql = " INSERT INTO orders (address_main, address_specific, order_date, payment, order_type, date_delivery, time_delivery, delivery_options, shipping_mode, shipping_fee, total, dedications, requests_details, order_status, customer_id)  
VALUES( '".$address_main."', '".$address_specific."','".$order_date."','".$payment."','".$order_type."','".$date_delivery."','".$time_delivery."','".$delivery_options."','".$shipping_mode."','".$shipping_fee."','".$total."','".$dedications."','".$requests_details."','".$order_status."', '".$customer_id."')    ";
 
if(mysqli_query($conn, $sql)){
    //echo "Success";
    $sql = "SELECT order_id FROM orders WHERE customer_id = '".$customer_id."'  AND order_date = '".$order_date."'";
    $result = mysqli_query($conn, $sql);
    $query_results = mysqli_fetch_array($result);
    $order_id= $query_results['order_id'];

}
else{
    //echo "Order database input error.";
}


//Cart table
if (!empty($_SESSION['cart'])) {
    foreach ($_SESSION['cart'] as $key => $value) {
        if ($value['customized']==1){
            //echo "importing new custom product.";

            $value['product_name'] = $firstName." ".$lastName." Customized ".$value['product_name'];

            $sql = "INSERT INTO products ( product_name, product_desc, price, cake_type, cake_size, cake_flavor, icing_flavor, customized, image_url)  
            VALUES ( '".$value['product_name']."', '".$value['product_desc']."', '".$value['price']."','".$value['cake_type']."','".$value['cake_size']."','".$value['cake_flavor']."','".$value['icing_flavor']."','".$value['customized']."', '".$value['image_url']."')";
            
            if(mysqli_query($conn, $sql)){
                //echo "Prodcut Import Success";
                $sql = "SELECT product_id FROM products WHERE product_name = '".$value['product_name']."'  AND image_url = '".$value['image_url']."'";
                $result = mysqli_query($conn, $sql);
                $query_results = mysqli_fetch_assoc($result);
                $product_id= $query_results['product_id'];
                            
            }
            else{
                //echo "Error in product import";
            }

            $price= $value['quantity'] * $value['price'];

            $sql = " INSERT INTO cart ( product_id, order_id, quantity,price)  
               VALUES ( '".$product_id."', '".$order_id."', '".$value['quantity']."','".$price."')";    
        }
        else{
            $price= $value['quantity'] * $value['price'];

            $sql = " INSERT INTO cart ( product_id, order_id, quantity,price)  
               VALUES ( '".$value['product_id']."', '".$order_id."', '".$value['quantity']."','".$price."')";    


        }
            
            if(mysqli_query($conn, $sql)){
                //echo "cart import Success";
               
                            
            }
            else{
                //echo "Error in cart import";
                   }       }}
else{
    //echo"Shopping Cart Empty.";

}
$_SESSION['reference_code']= $reference_in."O".strval($order_id);

header("Location: exit.php"); 
exit();

?>