-- phpMyAdmin SQL Dump
-- version 5.1.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: May 13, 2022 at 05:13 PM
-- Server version: 10.4.21-MariaDB
-- PHP Version: 8.0.11

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+08:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `kaye_bakes`
--

-- --------------------------------------------------------

--
-- Table structure for table `orders`
--

CREATE TABLE `products` (
  `product_id` int(15) NOT NULL,
  `product_name` varchar(20) NOT NULL,
  `product_desc` varchar(300),
  `price` int(11) NOT NULL,
  `type` varchar(15) NOT NULL,
  `cake_type` varchar(20),
  `cake_flavor` varchar(20),
  `icing_flavor` varchar(20),
  `deleted` tinyint(4) NOT NULL DEFAULT '0'
) ENGINE=InnoDB DEFAULT CHARSET=latin1; 


-- --------------------------------------------------------

-- Table structure for table `orders`


CREATE TABLE `orders` (
  `order_id`int(15) NOT NULL,
  `customer_id` int(15) NOT NULL,
  `address` varchar(300) NOT NULL,
  `date` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `payment` varchar(15) NOT NULL,
  `order_type` varchar(15) NOT NULL,
  `datetime_delivery` datetime NOT NULL,
  `delivery_options` varchar(20) NOT NULL,
  `shipping_fee` int(11) NOT NULL,
  `total` int(11) NOT NULL,
  `dedications` varchar(300) NOT NULL,
  `requests_details` varchar(300) NOT NULL,
  `deleted` tinyint(4) NOT NULL DEFAULT '0',
  `status` varchar(25) NOT NULL 
) ENGINE=InnoDB DEFAULT CHARSET=latin1;


-- --------------------------------------------------------

-- Table structure for table `cart`--

CREATE TABLE `cart` (
  `item_id` int(15) NOT NULL,
  `product_id` int(15) NOT NULL,
  `order_id` int(15) NOT NULL,
  `quantity` int(11) NOT NULL,
  `price` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;



-- --------------------------------------------------------

--
-- Table structure for table `customers`
--

CREATE TABLE `customers` (
  `customer_id`int(15) NOT NULL,
  `name` varchar(15) NOT NULL,
  `email` varchar(35) DEFAULT NULL,
  `address` varchar(300) DEFAULT NULL,
  `mobile` bigint(11) NOT NULL,
  `deleted` tinyint(4) NOT NULL DEFAULT '0'
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- ------------------------------------------------------
--
-- Indexes for table `products`
--
ALTER TABLE `products`
  ADD PRIMARY KEY (`product_id`);
--
-- Indexes for table `orders`
--
ALTER TABLE `orders`
  ADD PRIMARY KEY (`order_id`);

--
-- Indexes for table `cart`
--
ALTER TABLE `cart`
  ADD PRIMARY KEY (`item_id`);
  


-- Indexes for table `customers`
--
ALTER TABLE `customers`
  ADD PRIMARY KEY (`customer_id`);


-- --------------------------------------------------
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `items`
--
ALTER TABLE `products`
  MODIFY `product_id` int(15) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=100;

--
-- AUTO_INCREMENT for table `orders`
--
ALTER TABLE `orders`
  MODIFY `order_id` int(15) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=100;

--
-- AUTO_INCREMENT for table `cart`
--
ALTER TABLE `cart`
  MODIFY `item_id` int(15) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=1;


--
-- AUTO_INCREMENT for table `users`
--
ALTER TABLE `customers`
  MODIFY `customer_id` int(15) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=1;

-- ------------------------------------------------------------------------------------------
-- Constraints for dumped tables
--

--
-- Constraints for table `orders`
--
ALTER TABLE `orders`
  ADD CONSTRAINT `orders_ibfk_1` FOREIGN KEY (`customer_id`) REFERENCES `customers` (`customer_id`);

--
-- Constraints for table `order_details`
--
ALTER TABLE `cart`
  ADD CONSTRAINT `cart_ibfk_1` FOREIGN KEY (`product_id`) REFERENCES `products` (`product_id`),
  ADD CONSTRAINT `cart_ibfk_2` FOREIGN KEY (`order_id`) REFERENCES `orders` (`order_id`);

COMMIT;


-- ------------------------------------------------------------------------------------------
--
-- Menu Products
--

INSERT INTO `products` (`product_id`, `product_name`, `product_desc`, `price`, `Customized`, `image_url`, `cake_type`, `cake_flavor`, `icing_flavor`, `deleted`) VALUES
(27, 'Red Velvet Cupcakes', 'A dozen of soft, velvety cake with smooth cream cheese frosting and crushed Oreo cookies on top.', 430, 0, 'Red-Velvet-Cupcakes.jpg', NULL, NULL, NULL, 0),
(28, 'Dutch Chocolate Cupcakes', 'A dozen of moist, chocolate sponge cake highlighted by lemon icing on top.', 310, 0, 'Double-Dutch-Cupcakes.jpg', NULL, NULL, NULL, 0),
(29, 'Classic New York Cheesecake (small)', 'New York-style cheesecake with tasty and complementing crust. (6 inches)', 490, 0, 'Cheesecake(Small).jpg', NULL, NULL, NULL, 0),
(30, 'Cookies and Cream Cheesecake (Small)', 'New York cheesecake with Oreo cookies on its base, cake, and top. (6 inches)', 500, 0, 'Cookies-and-Cream-Cheesecake(small).jpg', NULL, NULL, NULL, 0),
(31, 'Cookies and Cream Cheesecake (Big)', 'New York cheesecake with Oreo cookies on its base, cake, and top. (6 inches)', 1000, 0, 'Cookies-and-Cream-Cheesecake(big).jpg', NULL, NULL, NULL, 0),
(32, 'Strawberry Cheesecake', 'New York cheesecake with a strawberry puree on top. (9 inches)', 1210, 0, 'Strawberry-Cheesecake(Big).jpg', NULL, NULL, NULL, 0),
(33, 'Blueberry Cheesecake', 'New York cheesecake with a Blueberry puree on top. (9 inches)', 1210, 0, 'Blueberry-Cheesecake(Big).jpg', NULL, NULL, NULL, 0),
(34, 'Oreo Cake (Small)', '3 layers of Oreo Chiffon Cake covered with Oreo Buttercream icing, drizzled with melted chocolate and topped with Oreos. (6 inches)', 650, 0, 'Oreo-Cake(small).jpg', NULL, NULL, NULL, 0),
(35, 'Oreo Cake (Big)', '3 layers of Oreo Chiffon Cake covered with Oreo Buttercream icing, drizzled with melted chocolate and topped with Oreos.(9 inches)', 1150, 0, 'Oreo-Cake(big).jpg', NULL, NULL, NULL, 0),
(36, 'Carrot Cake (Small)', 'A chunky Flavorful cake made with fresh carrots and real walnuts frosted with cream cheese icing. (9 inches)', 650, 0, 'Carrot-Cake(Small).jpg', NULL, NULL, NULL, 0),
(37, 'Carrot Cake (big)', 'A chunky Flavorful cake made with fresh carrots and real walnuts frosted with cream cheese icing. (9 inches)', 1050, 0, 'Carrot-Cake(big).jpg', NULL, NULL, NULL, 0),
(38, 'Strawberry Shortcake', 'Layers of Vanilla Chiffon filled with strawberry flavored buttercream, beautifully topped with fresh strawberries and florals. (9 inches)', 1200, 0, 'Strawberry-Shortcake.jpg', NULL, NULL, NULL, 0),
(39, 'Sans Rival (Small)', 'Layers of tender cashew Gateau filled with French Buttercream and more toasted cashews (6 inches)', 650, 0, 'Sans-Rival(small).jpg', NULL, NULL, NULL, 0),
(40, 'Sans Rival (big)', 'Layers of tender cashew Gateau filled with French Buttercream and more toasted cashews (9 inches)', 1350, 0, 'Sans-Rival(big).jpg', NULL, NULL, NULL, 0),
(41, 'Classic Crinkles', '60 Pieces of bite-sized crinkles loaded with chocolatey goodness.', 200, 0, 'Classic-Crinkles.jpg', NULL, NULL, NULL, 0),
(42, 'Red Velvet Crinkles', '15 sandwiches of delightful velvety crinkles filled with cream cheese frosting.', 265, 0, 'Red-Velvet-crinkles.jpg', NULL, NULL, NULL, 0),
(43, 'Double Choco Cookies', '15 chunky and chewy premium chocolate cookies loaded with chunks of dark chocolate.', 265, 0, 'Double-Dutch-Cookies.jpg', NULL, NULL, NULL, 0),
(44, 'Smores Bars', '15 Smore bars made with thick, tasty crust, premium chocolate, toasted mallows and Hershey chocolate on top.', 265, 0, 'Smores.jpg', NULL, NULL, NULL, 0);

--
-- Menu Products
--
-- ------------------------------------------------------------------------------------------
--
-- Custom Products
--

INSERT INTO `products`(`product_name`, `product_desc`, `price`, `Customized`, `image_url`, `cake_type`, `cake_flavor`, `icing_flavor`, `deleted`, `cake_size`) 

VALUES 
('Octonauts-themed Cake','From the famous television series, "Octonauts" that started in 2010, we are inspired to showcase Captain Barnacles and his team mates','800','1','202417718_325487462540206_3769384308967751756_n.jpg','cake','chocolate','0','fondant','6inches'),
('Overloaded Oreo Cake','3 layers of Oreo Chiffon Cake covered with Oreo Buttercream icing, drizzled with melted chocolate and topped with Oreos.','1150','1','190467260_306534187768867_5484906304633274511_n.jpg','cake','chocolate','buttercream','0','9inches'),
('Carrot Cake','A chunky Flavorful cake made with fresh carrots and real walnuts frosted with cream cheese icing.','1050','1','272083068_463967432025541_5798129019754268316_n.jpg','cake','carrot','cream_cheese','0','9inches'),
('Chocolate Cake','Chocolate Cake covered with Buttercream Icing','375','1','271628654_459609585794659_3906463957166714636_n.jpg','cake','chocolate','buttercream','0','4inches'),
(' Chocolate Cake','Chocolate Cake covered with Buttercream Icing','490','1','260287185_430790272009924_1192468824419506728_n.jpg','cake','chocolate','buttercream','0','6inches'),
('Red Velvet Cake','Behind this buttercream icing, hides our moist and flavorful red velvet signature cake.','650','1','244178594_395318258890459_2987269854333336011_n.jpg','cake','red_velvet','buttercream','0','6inches'),
('Money Cake','Made with mango chiffon, then filled with fresh mangoes and vegan whipped cream.','850','1','242936400_390876266001325_1363715095845968859_n.jpg','cake','mango_chiffon','merengue','0','6inches'),
('Red Velvet Cake','Behind this buttercream icing, hides our moist and flavorful red velvet signature cake.','1050','1','242373253_388745472881071_7295517219885017801_n.jpg','cake','red_velvet','buttercream','0','9inches'),
(' Sansrival Cake','This is layers of cashew wafer and french buttercream stacked perfectly for a satisfying indulgence.','1350','1','120493751_167107881711499_2655423482051305365_n.jpg','cake','sans_rival','buttercream','0','9inches'),
('Chocolate Money Cake',"If you desire for a money cake, then we're here to make it as far as we can, with this Chocolate Cake topped with Buttercream Icing.",'750','1','225771459_351762159912736_6439062695661552583_n.jpg','cake','chocolate','buttercream','0','6inches'),
('Red Velvet Money Cake',"Because of our gratitude to them everyday, it will never be too late to feature this wonderful red velvet money cake with buttercream icing for Mother's day",'1350','1','213335771_336573321431620_8076832528081215742_n.jpg','cake','red_velvet','buttercream','0','6inches'),
('Mocha Chiffon Fondant Cake','Send love and hugs to our doctors, nurses and our frontliners through our original and freshly baked Mocha Chiffon with Fondant Cake','950','1','173935308_283294686759484_7592430548018566313_n.jpg','cake','mocha_chiffon','fondant','0','6inch'),
('Vanilla Chiffon','Vanilla Chiffon with Buttercream Frosting','650','1','164576745_272088754546744_7819501716608156782_n.jpg','cake','vanila_chiffon','buttercream','0','6inches'),
('Red Velvet Cake','Red Velvet Cake with Cream Cheese Frosting','1050','1','164742965_269373644818255_2170657300178414259_n.jpg','cake','red_velvet','cream_cheese','0','9inches'),
('Chocolate Money Cake',"If you desire for a money cake, then we're here to make it as far as we can, with this Chocolate Cake topped with Buttercream Icing.",'1100','1','243517490_394065989015686_4040078969147043719_n.jpg','cake','chocolate','buttercream','0','9inches'),
('Totoro X Strawberry Cake','Layers of Vanilla Chiffon filled with strawberry flavored buttercream, beautifully topped with fresh strawberries.','1200','1','196054456_317206083368344_559117140705000238_n.jpg','cake','strawberry','buttercream','0','9inches'),
('Choco Buttercream','Chocolate cake with buttercream frosting','490','1','167910179_276653607423592_3783542180133165779_n.jpg','cake','chocolate','buttercream','0','6inches'),
('Red velvet money cake',"if you desire for a money cake, then we're here to make it as far as we can with our Red velvet money cake with Cream cheese frosting.",'1450','1','193987349_314505813638371_3286078867865827075_n.jpg','cake','red_velvet','buttercream','0','9inches'),
('Strawberry Shortcake with Flowers','Layers of Vanilla Chiffon filled with strawberry flavored buttercream, beautifully topped with fresh strawberries and florals.','1200','1','173380409_283938260028460_8385848213773839715_n.jpg','cake','strawberry','buttercream','0','9inches'),
('Chocolate Money Cake',"If you desire for a money cake, then we're here to make it as far as we can, with this Chocolate Cake topped with Buttercream frosting.",'1100','1','160402946_261447985610821_7715676171921234966_n.jpg','cake','chocolate','buttercream','0','9inches'),
('Carrot Cake','Designed as one of our healthiest desserts, our carrot cake features moist, balance and flavor','1050','1','150102913_246723707083249_8020259760178085492_n.jpg','cake','carrot','cream_cheese','0','9inch'),
('Chocolate cake with Blue Velvet Crinkles','Craving to add a customized cake with our crinkles? here is a Chocolate cake with Buttercream frosting with Blue Velvet Crinkles','1200','1','199941638_322939712794981_1450613446828329090_n.jpg','cake','chocolate','buttercream','0','6inches'),
('Red Velvet Cake','Designed with love, this sunflower cake is beyond heartbeats with its red velvet flavor covered with buttercream frosting.','1050','1','133204548_214499756972311_1153710032686718458_n.jpg','cake','red_velvet','cream_cheese','0','9inches'),
('Choco Buttercream Cake','Snow ice-themed cake with a chocolate base and buttercream icing.','490','1','128349408_192418462513774_6598218976123063567_n.jpg','cake','chocolate','buttercream','0','6inches'),
('Choco Buttercream Heart-shaped Cake',"It's a Choco Buttercream Heart-shaped Cake drizzled with blue-colored white Chocolate",'450','1','122208214_174226874332933_6200804887330506532_n.jpg','cake','chocolate','buttercream','0','4inches'),
('Carrot Cake with Fresh Fruits','A chunky Flavorful cake made with fresh carrots, topped with fresh tropical fruits','850','1','188965631_309742320781387_2686826666501265164_n.jpg','cake','carrot','cream_cheese','0','9inch'),
('Chocolate heart-shaped cake','Heart-shaped cake with a Chocolate base and Buttercream frosting','376','1','178457719_291881782567441_129465566001969973_n.jpg','cake','chocolate','buttercream','0','4inches'),
('Semi fondant chocolate cake','A 4-inch chocolate cake with a buttercream frosting and fondant at the base','450','1','177470358_291881142567505_8526036144150620132_n.jpg','cake','chocolate','buttercream','0','4inches'),
('Choco Buttercream','Chocolate cake with buttercream frosting','490','1','180593871_295870385501914_7495208871144748451_n.jpg','cake','chocolate','buttercream','0','6inches');

--
-- Custom Products
--
-- ------------------------------------------------------------------------------------------

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;