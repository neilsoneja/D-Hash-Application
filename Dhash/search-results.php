<?php
session_start();
//retrieving query imagae filename 
$queryName = $_SESSION['query-filename'];

//resseting output array
$output=array();

//executing python Dhash search algo
$output = exec( "python3 search.py --q uploads/".$queryName);

//transferring search results to paths array
$paths = array();
  foreach ($output as $filename){
      //path to images dir
      $path="/Kaye-Bakes-2/account/Dhash/images/".$filename;
      $paths[]=$path; 
  }

//path to inquiry image
$queryPath="/Kaye-Bakes-2/account/Dhash/uploads/".$queryName;

?>
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="X-UA-Compatible" content="IE=edge">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>Kaye Bakes | Customized Order</title>

<link rel="stylesheet" type="text/css" href="/Kaye-Bakes-2/account/css/bootstrap.min.css">
<link rel="stylesheet" type="text/css" href="/Kaye-Bakes-2/account/css/font-awesome.min.css">
<link rel="stylesheet" type="text/css" href="/Kaye-Bakes-2/account/css/bistro-icons.css">
<link rel="stylesheet" type="text/css" href="/Kaye-Bakes-2/account/css/settings.css">
<link rel="stylesheet" type="text/css" href="/Kaye-Bakes-2/account/css/owl.carousel.css">
<link rel="stylesheet" type="text/css" href="/Kaye-Bakes-2/account/css/owl.transitions.css">
<link rel="stylesheet" type="text/css" href="/Kaye-Bakes-2/account/css/owl.carousel.css">
<link rel="stylesheet" type="text/css" href="/Kaye-Bakes-2/account/css/jquery.fancybox.css">
<link rel="stylesheet" type="text/css" href="/Kaye-Bakes-2/account/css/zerogrid.css">
<link rel="stylesheet" type="text/css" href="/Kaye-Bakes-2/account/css/style.css">
<link rel="stylesheet" type="text/css" href="/Kaye-Bakes-2/account/css/loader.css">
<link rel="shortcut icon" href="images/favicon.png">

<!--[if lt IE 9]>
    <script src="https://oss.maxcdn.com/html5shiv/3.7.2/html5shiv.min.js"></script>
    <script src="https://oss.maxcdn.com/respond/1.4.2/respond.min.js"></script>
<![endif]-->

</head>



<body>

<!--Loader-->
<div class="loader"> 
   <div class="cssload-container">
     <div class="cssload-circle"></div>
     <div class="cssload-circle"></div>
   </div>
</div>

<!--Top bar-->
<div class="topbar">
  <div class="container">
    <div class="row">
      <div class="col-md-12">
        <p class="pull-left hidden-xs">Kaye Bakes Bakeshop</p>
        <p class="pull-right"><i class="fa fa-ambulance"></i>Order Online (+63) 935 318 3517</p>
      </div>
    </div>
  </div>
</div>


<header id="main-navigation">
  <div id="navigation" data-spy="affix" data-offset-top="20">
    <div class="container">
      <div class="row">
      <div class="col-md-12">
        <nav class="navbar navbar-default">
          <div class="navbar-header">
            <button type="button" class="navbar-toggle collapsed" data-toggle="collapse" data-target="#fixed-collapse-navbar" aria-expanded="false"> 
            <span class="icon-bar top-bar"></span> <span class="icon-bar middle-bar"></span> <span class="icon-bar bottom-bar"></span> 
            </button>
           <a class="navbar-brand" href="index.html"><img src="/Kaye-Bakes-2/account/images/logo1.png" alt="logo" class="img-responsive"></a> 
         </div>
        
         <div id="fixed-collapse-navbar" class="navbar-collapse collapse navbar-right">
          <ul class="nav navbar-nav">
            <li>
               <a href="index.html">Home</a>
               
            </li>
           <!-- <li><a href="food.html">Our Food</a></li>
            
            
                <li><a href="about.html">About Us</a></li>
                <li><a href="faq.html">FAQ</a></li>-->
              
            <li><a href="./account/register.php">Order Now</a></li>
            <li><a href="gallery.html">Gallery</a></li>
            <li><a href="about.html">About US</a></li>
            
                  </ul>
                </li>
              </ul>
            </div>
         </nav>
         </div>
       </div>
     </div>
  </div>
</header>

<!--Page header & Title
<section id="page_header">
<div class="page_title">
  <div class="container">
    <div class="row">
      <div class="col-md-12">
        
      </div>
    </div>
  </div>
</div>  
</section>-->


<section id="gallery" class="padding-top">
  <div class="container">
    <!--query image-->
  <div class="row">
      <div class="zerogrid">
        <div class="wrap-container">
          <div class="wrap-content clearfix">

        <div class="col-1-3 mix work-item cakes heading_space">
              <div class="wrap-col">
                <div class="item-container">
                  <div class="image">
                    <img src="<?php echo $queryPath; ?>" alt="cook"/>
                    <div class="overlay">
                        <a class="fancybox overlay-inner" href="<?php echo $queryPath; ?>" data-fancybox-group="gallery"><i class=" icon-eye6"></i></a>
                    </div>
                  </div> 
                  <div class="gallery_content">
                    <h3><a href="recepi_detail.html"> Uploaded Image</a></h3>
                    <p>desc</p>
                  </div>
                </div>
              </div>
            </div>
            
          </div>
         </div>
      </div>
   </div>
  <!--search results-->
     <div class="row">
      <div class="zerogrid">
        <div class="wrap-container">
          <div class="wrap-content clearfix">

              <!--Row 1-1 -->          
              <div class="col-1-3 mix work-item cakes heading_space">
              <div class="wrap-col">
                <div class="item-container">
                  <div class="image">
                    <img src="<?php echo $paths[0]; ?>" alt="cook"/>
                    <div class="overlay">
                        <a class="fancybox overlay-inner" href="<?php echo $paths[0]; ?>" data-fancybox-group="gallery"><i class=" icon-eye6"></i></a>
                    </div>
                  </div> 
                  <div class="gallery_content">
                    <h3><a href="recepi_detail.html">link</a></h3>
                    <p>desc</p>
                  </div>
                </div>
              </div>
            </div>
                <!--Row 1-2 -->          
                <div class="col-1-3 mix work-item cakes heading_space">
              <div class="wrap-col">
                <div class="item-container">
                  <div class="image">
                    <img src="<?php echo $paths[1]; ?>" alt="cook"/>
                    <div class="overlay">
                        <a class="fancybox overlay-inner" href="<?php echo $paths[1]; ?>" data-fancybox-group="gallery"><i class=" icon-eye6"></i></a>
                    </div>
                  </div> 
                  <div class="gallery_content">
                    <h3><a href="recepi_detail.html"> link</a></h3>
                    <p>desc</p>
                  </div>
                </div>
              </div>
            </div>
          <!--Row 1-3 -->          
          <div class="col-1-3 mix work-item cakes heading_space">
              <div class="wrap-col">
                <div class="item-container">
                  <div class="image">
                    <img src="<?php echo $paths[2]; ?>" alt="cook"/>
                    <div class="overlay">
                        <a class="fancybox overlay-inner" href="<?php echo $paths[2]; ?>" data-fancybox-group="gallery"><i class=" icon-eye6"></i></a>
                    </div>
                  </div> 
                  <div class="gallery_content">
                    <h3><a href="recepi_detail.html"> link</a></h3>
                    <p>desc</p>
                  </div>
                </div>
              </div>
            </div>
                <!--Row 2-1 -->          
                <div class="col-1-3 mix work-item cakes heading_space">
              <div class="wrap-col">
                <div class="item-container">
                  <div class="image">
                    <img src="<?php echo $paths[3]; ?>" alt="cook"/>
                    <div class="overlay">
                        <a class="fancybox overlay-inner" href="<?php echo $paths[3]; ?>" data-fancybox-group="gallery"><i class=" icon-eye6"></i></a>
                    </div>
                  </div> 
                  <div class="gallery_content">
                    <h3><a href="recepi_detail.html">link</a></h3>
                    <p>desc</p>
                  </div>
                </div>
              </div>
            </div>
                <!--Row 2-2 -->          
                <div class="col-1-3 mix work-item cakes heading_space">
              <div class="wrap-col">
                <div class="item-container">
                  <div class="image">
                    <img src="<?php echo $paths[4]; ?>" alt="cook"/>
                    <div class="overlay">
                        <a class="fancybox overlay-inner" href="<?php echo $paths[4]; ?>" data-fancybox-group="gallery"><i class=" icon-eye6"></i></a>
                    </div>
                  </div> 
                  <div class="gallery_content">
                    <h3><a href="recepi_detail.html">link</a></h3>
                    <p>desc</p>
                  </div>
                </div>
              </div>
            </div>
                <!--Row 2-3 -->          
                <div class="col-1-3 mix work-item cakes heading_space">
              <div class="wrap-col">
                <div class="item-container">
                  <div class="image">
                    <img src="<?php echo $paths[5]; ?>" alt="cook"/>
                    <div class="overlay">
                        <a class="fancybox overlay-inner" href="<?php echo $paths[5]; ?>" data-fancybox-group="gallery"><i class=" icon-eye6"></i></a>
                    </div>
                  </div> 
                  <div class="gallery_content">
                    <h3><a href="recepi_detail.html">link</a></h3>
                    <p>desc</p>
                  </div>
                </div>
              </div>
            </div>
                <!--Row 3-1 -->          
                <div class="col-1-3 mix work-item cakes heading_space">
              <div class="wrap-col">
                <div class="item-container">
                  <div class="image">
                    <img src="<?php echo $paths[6]; ?>" alt="cook"/>
                    <div class="overlay">
                        <a class="fancybox overlay-inner" href="<?php echo $paths[6]; ?>" data-fancybox-group="gallery"><i class=" icon-eye6"></i></a>
                    </div>
                  </div> 
                  <div class="gallery_content">
                    <h3><a href="recepi_detail.html">link</a></h3>
                    <p>desc</p>
                  </div>
                </div>
              </div>
            </div>
                <!--Row 3-2 -->          
                <div class="col-1-3 mix work-item cakes heading_space">
              <div class="wrap-col">
                <div class="item-container">
                  <div class="image">
                    <img src="<?php echo $paths[7]; ?>" alt="cook"/>
                    <div class="overlay">
                        <a class="fancybox overlay-inner" href="<?php echo $paths[7]; ?>" data-fancybox-group="gallery"><i class=" icon-eye6"></i></a>
                    </div>
                  </div> 
                  <div class="gallery_content">
                    <h3><a href="recepi_detail.html">link</a></h3>
                    <p>desc</p>
                  </div>
                </div>
              </div>
            </div>
                <!--Row 3-3 -->          
                <div class="col-1-3 mix work-item cakes heading_space">
              <div class="wrap-col">
                <div class="item-container">
                  <div class="image">
                    <img src="<?php echo $paths[8]; ?>" alt="cook"/>
                    <div class="overlay">
                        <a class="fancybox overlay-inner" href="<?php echo $paths[8]; ?>" data-fancybox-group="gallery"><i class=" icon-eye6"></i></a>
                    </div>
                  </div> 
                  <div class="gallery_content">
                    <h3><a href="recepi_detail.html">link</a></h3>
                    <p>desc</p>
                  </div>
                </div>
              </div>
            </div>
           <!--------------------------------> 
          </div>
        </div>
      </div>
     </div>
  </div>
</section>



<!--Footer-->
<footer class="padding-top bg_black">
  <div class="container">
    <div class="row">
      <div class="col-md-3 col-sm-6 footer_column">
        <h4 class="heading">Who are we?</h4>
        <hr class="half_space">
        <p class="half_space">Kaye Bakes is a cake shop that was founded by Kaye Santillan out of her passion and love for baking. She started selling brownies and other pastries as a student in 2012. Experience from her work as a barista and further baking courses led her to start her own baking shop. </p>
        <p>In 2016, Kaye Bakes was officially launched and branded. They mainly operated in the Cavite area and they continued to expand their network of customers and the complexity of their menu for four year. The shop was revamped in 2020 to what it is today.</p>
      </div>
      <div class="col-md-3 col-sm-6 footer_column">
        <h4 class="heading">Quick Links</h4>
        <hr class="half_space">
        <ul class="widget_links">
          <li><a href="/Kaye-Bakes-2/index.html">Home</a></li>
          <li><a href="about.html">About Us</a></li>
          <li><a href="order.html">Order Now</a></li>
          <li><a href="gallery.html">Gallery</a></li>
        </ul>
      </div>
      <div class="col-md-3 col-sm-6 footer_column">
        <h4 class="heading">Newsletter</h4>
        <hr class="half_space">
        <p class="icon"><i class="icon-dollar"></i>Sign up with your name and email to get updates fresh updates.</p>
        <div id="result1" class="text-center"></div>        
        
       <form action="http://themesindustry.us13.list-manage.com/subscribe/post-json?u=4d80221ea53f3a4487ddebd93&id=494727d648&c=?" method="get" onSubmit="return false" class="newsletter">
          <div class="form-group">
            <input type="email" placeholder="E-mail Address" required name="EMAIL" id="EMAIL" class="form-control" />
          </div>
          <div class="form-group">
            <input type="submit" class="btn-submit button3" value="Subscribe" />
          </div>
        </form>
      </div>
      <div class="col-md-3 col-sm-6 footer_column">
        <h4 class="heading">Get in Touch</h4>
        <hr class="half_space">
        <p></p>
        <p class="address"><i class="icon-location"></i>Kaye Bakes Bakeshop, Wet & Dry Public Market, Governor's Dr, Trece Martires, Cavite</p>
        <p class="address"><i class="fa fa-phone"></i>(+63) 935 318 3517</p>

      </div> 
    </div>
    <div class="row">
     <div class="col-md-12">
        <div class="copyright clearfix">
          <p>Copyright &copy; 2016 Kaye Bakes. All Right Reserved</p>
          <ul class="social_icon">
               <li><a href="#." class="facebook"><i class="icon-facebook5"></i></a></li>
               <li><a href="#." class="twitter"><i class="icon-twitter4"></i></a></li>
               <li><a href="#." class="google"><i class="icon-google"></i></a></li>
              </ul>
        </div>
      </div>
    </div>
  </div>
</footer>

<a href="#." id="back-top"><i class="fa fa-angle-up fa-2x"></i></a>
<script src="/Kaye-Bakes-2/account/js/jquery-2.2.3.js" type="text/javascript"></script>
<script src="/Kaye-Bakes-2/account/js/bootstrap.min.js" type="text/javascript"></script>
<script src="/Kaye-Bakes-2/account/js/jquery.parallax-1.1.3.js"></script>
<script src="/Kaye-Bakes-2/account/js/jquery.appear.js"></script>  
<script src="/Kaye-Bakes-2/account/js/jquery-countTo.js"></script>
<script src="/Kaye-Bakes-2/account/js/owl.carousel.min.js" type="text/javascript"></script>
<script src="/Kaye-Bakes-2/account/js/jquery.fancybox.js"></script>
<script src="/Kaye-Bakes-2/account/js/jquery.mixitup.min.js"></script>
<script src="/Kaye-Bakes-2/account/js/functions.js" type="text/javascript"></script>
<script src="/Kaye-Bakes-2/account/js/masonry.pkgd.min.js" type="text/javascript"></script>
</body>
</html>
