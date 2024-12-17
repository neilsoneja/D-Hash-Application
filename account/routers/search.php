<?php
$dbservername = "localhost";
$dbUsername = "root";
$dbPassword = "";
$dbName = "kaye-bakes";
$conn = mysqli_connect($dbservername,$dbUsername,$dbPassword,$dbName);

if (isset($_POST['submit-search'])) {
    $search = mysqli_real_escape_string($conn, $_POST['search']);
    $sql = "SELECT * FROM products WHERE product_name LIKE '%search%' or product_desc LIKE '%search%'";
    $result = mysqli_query($conn, $sql);
    $query_results = mysqli_num_rows($result);

    if ($query_results > 0) {
        while ($row = mysqli_fetch_assoc($result)) {
          echo 
          "<a href='product.php?title=".$row['product_name']."&image=".$row['image_url']."'>
          <div class='product-box'>
          <img src='product_images/".$row['image_url']."' >
          <h3>".$row['product_name']."</h3>
          <p hidden>".$row['product_desc']."</p>
          <p>Php ".$row['price']."</p>
          <button type='submit' name='add-to-cart'>Add to Cart</button>
          </div>
          </a>";
        }
    } else {
        echo "No results";
    }
}
?>