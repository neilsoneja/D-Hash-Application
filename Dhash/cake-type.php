<?php
session_start();

?>
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="description" content="">
    <title>customize</title>

    <link rel="stylesheet" type="text/css"  media="screen,projection" href="/Kaye-Bakes-2/account/css/new.css">
    <link rel="stylesheet" type="text/css" href="/Kaye-Bakes-2/account/css/bootstrap/bootstrap.min.css">
    <link rel="stylesheet" type="text/css" href="/Kaye-Bakes-2/account/css/font-awesome.min.css">
    <link rel="stylesheet" type="text/css" href="/Kaye-Bakes-2/account/css/features.css">
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

      .centered {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
      }

    .shadow {
    text-shadow:  2px 2px 5px #070707;
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

  <section class="pt-5  container ">
  
    <div class="row ">

      <div class="bebas col text-center">
        <h1 class=" display-3">Cake Selection </h1>
      </div>
    </div>
    <hr class="my-4 ">
    <p class=" fs-4 text-center">Choose the cake type for your customized cake</p>
    <hr class="my-4 ">
  </section>
  
    



  
  <div class="album py-5 bg-transparent" >
    <div class="container">

      <div class="row row-cols-1 row-cols-sm-2 row-cols-md-3 g-3">
        
        <form method="post" action="/Kaye-Bakes-2/account/customize-order-new.php">
        <div class="list-group list-group-horizontal p-0 m-0">
          <input class="list-group-item-check pe-none "  name="cake_type" id="cake_type" value="bento" >
          <label class=" border-0 list-group-item rounded-3 py-3 " for="bento">
            <button type="submit"  class="border-0 bg-transparent">
            <img src="/Kaye-Bakes-2/account/Dhash/images/new/bento@/278223523_304469101835593_8057979303448465191_n.jpg" class="img-fluid rounded-4" alt="...">
            <div class="sailyme centered">
              <h1 class=" pt-1 text-center rounded-3 display-5 shadow  pt-5 text-light" >  Bento </h1>
             </div>  
          </button>
          </label>
        </div>
      </form>
      <form method="post" action="/Kaye-Bakes-2/account/customize-order-new.php">

        <div class="list-group list-group-horizontal  p-0 m-0 ">
          <input class="list-group-item-check pe-none"  name="cake_type" id="cake_type" value="cake">
          <label class="border-0 list-group-item rounded-3 py-3" for="cake">
            <button type="submit"  class="border-0 bg-transparent">
            <img src="/Kaye-Bakes-2/account/Dhash/images/new/cake@/242373253_388745472881071_7295517219885017801_n.jpg" class="img-fluid rounded-4" alt="...">
            <div class="sailyme centered">
             <h1 class="pt-1 text-center rounded-3 display-5 shadow  pt-5 text-light">  Cake </h1>
            </div>
          </button>
          </label>
        </div>
      </form>


      <form method="post" action="/Kaye-Bakes-2/account/customize-order-new.php">
        <div class="list-group list-group-horizontal   p-0 m-0">
          <input class="list-group-item-check pe-none"  name="cake_type" id="cake_type" value="cat">
          <label class="border-0 list-group-item rounded-3 py-3" for="cat">
            <button type="submit"  class="border-0 bg-transparent">
            <img src="/Kaye-Bakes-2/account/Dhash/images/new/cat@/186897567_304510991304520_2785647618921207653_n.jpg" class="img-fluid rounded-4" alt="...">

            <div class="sailyme centered">
             <h1 class="pt-1 text-center rounded-3 display-5 shadow  pt-5 text-light">  Cat Cake</h1>
            </div>
          </button>
          </label>
        </div>

      </form>
      <form method="post" action="/Kaye-Bakes-2/account/customize-order-new.php">
        <div class="list-group list-group-horizontal p-0 m-0">
          <input class="list-group-item-check pe-none "  name="cake_type" id="cake_type" value="cupcakes" >
          <label class=" border-0 list-group-item rounded-3 py-3 " for="cupcakes">
            <button type="submit"  class="border-0 bg-transparent">
            <img src="/Kaye-Bakes-2/account/Dhash/images/new/cupcakes@/223650896_351836436571975_450125279364246809_n.jpg" class="img-fluid rounded-4" alt="...">

            <div class="sailyme centered">
             <h1 class="pt-1 text-center rounded-3 display-5 shadow  pt-5 text-light">  Cupcakes </h1>
            </div>
          </button>
          </label>
        </div>
      </form>
      <form method="post" action="/Kaye-Bakes-2/account/customize-order-new.php">
       <div class="list-group list-group-horizontal p-0 m-0">
          <input class="list-group-item-check pe-none "  name="cake_type" id="cake_type" value="dog" >
          <label class=" border-0 list-group-item rounded-3 py-3 " for="dog">
            <button type="submit"  class="border-0 bg-transparent">
            <img src="/Kaye-Bakes-2/account/Dhash/images/new/dog@/243348065_393376499084635_3518146004564107815_n.jpg" class="img-fluid rounded-4" alt="...">

            <div class="sailyme centered">
             <h1 class="pt-1 text-center rounded-3 display-5 shadow  pt-5 text-light"> Dog Cakes</h1>
            </div>
          </button>
          </label>
       </div>
      </form>
      <form method="post" action="/Kaye-Bakes-2/account/customize-order-new.php">
        <div class="list-group list-group-horizontal p-0 m-0">
          <input class="list-group-item-check pe-none "  name="cake_type" id="cake_type" value="layered" >
          <label class=" border-0 list-group-item rounded-3 py-3 " for="layered">
            <button type="submit"  class="border-0 bg-transparent">
            <img src="/Kaye-Bakes-2/account/Dhash/images/new/layered@/260287185_430790272009924_1192468824419506728_n.jpg" class="img-fluid rounded-4" alt="...">

            <div class="sailyme centered">
             <h1 class="pt-1 text-center rounded-3 display-5 shadow  pt-5 text-light"> Layered Cake</h1>
            </div>
          </button>
          </label>
        </div>
      </form>
      <form method="post" action="/Kaye-Bakes-2/account/customize-order-new.php">
        <div class="list-group list-group-horizontal p-0 m-0">
          <input class="list-group-item-check pe-none "  name="cake_type" id="cake_type" value="money" >
          <label class=" border-0 list-group-item rounded-3 py-3 " for="money">
            <button type="submit"  class="border-0 bg-transparent">
            <img src="/Kaye-Bakes-2/account/Dhash/images/new/money@/2164576745_272088754546744_7819501716608156782_n.jpg" class="img-fluid rounded-4" alt="...">

            <div class="sailyme centered">
             <h1 class="pt-1 text-center rounded-3 display-5 shadow  pt-5 text-light">  Money Cake </h1>
            </div>
          </button>
          </label>
        </div>

      </form>
      <form method="post" action="/Kaye-Bakes-2/account/customize-order-new.php">
        <div class="list-group list-group-horizontal p-0 m-0">
          <input class="list-group-item-check pe-none "  name="cake_type" id="cake_type" value="number" >
          <label class=" border-0 list-group-item rounded-3 py-3 " for="number">
            <button type="submit"  class="border-0 bg-transparent">
            <img src="/Kaye-Bakes-2/account/Dhash/images/new/number@/173116762_281318236957129_7119905468423422376_n.jpg" class="img-fluid rounded-4" alt="...">

            <div class="sailyme centered">
             <h1 class="pt-1 text-center rounded-3 display-5 shadow  pt-5 text-light">  Number Cake </h1>
            </div>
          </button>
          </label>
        </div>

      </form>
      <form method="post" action="/Kaye-Bakes-2/account/customize-order-new.php">
        <div class="list-group list-group-horizontal p-0 m-0">
          <input class="list-group-item-check pe-none "  name="cake_type" id="cake_type" value="number_cup" >
          <label class=" border-0 list-group-item rounded-3 py-3 " for="number_cup">
            <button type="submit"  class="border-0 bg-transparent">
            <img src="/Kaye-Bakes-2/account/Dhash/images/new/number_cup@/199521318_319782549777364_2382785519519199026_n.jpg" class="img-fluid rounded-4" alt="...">

            <div class="sailyme centered">
             <h1 class="pt-1 text-center rounded-3 display-5 shadow  pt-5 text-light"> Number Cupcake</h1>
            </div>
          </button>
          </label>
        </div>
    
      </form>


        <div class="list-group list-group-horizontal p-0 m-0">
          <input class="list-group-item-check pe-none "  >
          <label class=" border-0 list-group-item rounded-3 py-3 " for="wedding">

            <div class="sailyme centered">
             <h1 class="pt-1 text-center rounded-3 display-5 shadow  pt-5 text-light" > </h1>
            </div>
          </label>
        </div>
      

      <form method="post" action="/Kaye-Bakes-2/account/customize-order-new.php">
        <div class="list-group list-group-horizontal p-0 m-0">
          <input class="list-group-item-check pe-none "  name="cake_type" id="cake_type" value="wedding" >
          <label class=" border-0 list-group-item rounded-3 py-3 " for="wedding">
            <button type="submit"  class="border-0 bg-transparent">
            <img src="/Kaye-Bakes-2/account/Dhash/images/new/wedding@/133731692_214497856972501_5285741356268823605_n.jpg" class="img-fluid rounded-4" alt="...">

            <div class="sailyme centered">
             <h1 class="pt-1 text-center rounded-3 display-5 shadow  pt-5 text-light" >Wedding Cake </h1>
            </div>
          </button>
          </label>
        </div>
      </form>


      </div>
   </div> 
  </div>





</main>

<footer class="text-muted py-5">
  <div class="container">
    <p class="float-end mb-1">
      <a href="#">Back to top</a>
    </p>
    <p class="mb-1">Album example is &copy; Bootstrap, but please download and customize it for yourself!</p>
    <p class="mb-0">New to Bootstrap? <a href="/">Visit the homepage</a> or read our <a href="../getting-started/introduction/">getting started guide</a>.</p>
  </div>
</footer>

 <!--bootstrap-->
 <script type="text/javascript" src="/Kaye-Bakes-2/account/js/bootstrap/bootstrap.bundle.min.js"></script>  

<!-- jQuery Library -->
<script type="text/javascript" src="/Kaye-Bakes-2/account/js/jquery-3.4.1.min.js"></script>

<!--bootstrap-->
<script type="text/javascript" src="/Kaye-Bakes-2/account/js/bootstrap/bootstrap-datepicker.min.js"></script>

<!--custom-->
<script type="text/javascript" src="/Kaye-Bakes-2/account/js/own-script.js"></script>


 <!--loader-->
 <script src="js/functions.js" type="text/javascript"></script>

<script>  
function preview() {
    frame.src = URL.createObjectURL(event.target.files[0]);
}
function clearImage() {
    document.getElementById('formFile').value = null;
    frame.src = "";
}


 
$(window).on('load', function () {
    $('#loading').fadeOut("slow");
  }) 

</script>    
  </body>
</html>
