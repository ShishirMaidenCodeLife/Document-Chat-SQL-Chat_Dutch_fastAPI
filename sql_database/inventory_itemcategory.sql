-- phpMyAdmin SQL Dump
-- version 5.2.0
-- https://www.phpmyadmin.net/
--
-- Host: localhost
-- Generation Time: Dec 05, 2023 at 10:50 AM
-- Server version: 10.4.27-MariaDB
-- PHP Version: 8.2.0

-- SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
-- START TRANSACTION;
-- SET time_zone = "+00:00";


-- /*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
-- /*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
-- /*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
-- /*!40101 SET NAMES utf8mb4 */;

--
-- Database: `inventory`
--

-- --------------------------------------------------------

--
-- Table structure for table `inventory_itemcategory`
--

CREATE TABLE `inventory_itemcategory` (
  `id` bigint(20) NOT NULL,
  `category_name` varchar(50) DEFAULT NULL,
  `created_by` int(11) DEFAULT NULL,
  `organization_id` int(11) DEFAULT NULL,
  `updated_at` datetime(6) NOT NULL,
  `created_at` datetime(6) NOT NULL
  )
-- ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `inventory_itemcategory`
--

INSERT INTO `inventory_itemcategory` (`id`, `category_name`, `created_by`, `organization_id`, `updated_at`, `created_at`) VALUES
(1, 'menu', 1, 1, '2023-12-03 12:15:03.209761', '2023-12-03 12:15:03.209328'),
(2, 'A', 1, 1, '2023-12-03 14:11:15.509865', '2023-12-03 14:11:15.509668'),
(3, 'B', 1, 1, '2023-12-03 14:13:49.651146', '2023-12-03 14:13:49.650926'),
(5, 'laptop', 1, 2, '2023-12-05 10:57:10.113322', '2023-12-05 10:57:10.113101'),
(6, 'pc', 1, 1, '2023-12-05 11:03:47.190049', '2023-12-05 11:03:47.189922'),
(7, 'laptop', 1, 1, '2023-12-05 12:00:49.073863', '2023-12-05 12:00:49.073649'),
(8, 'burger', 1, 1, '2023-12-05 13:26:29.897530', '2023-12-05 13:26:29.897242');

-- --
-- -- Indexes for dumped tables
-- --

-- --
-- -- Indexes for table `inventory_itemcategory`
-- --
-- ALTER TABLE `inventory_itemcategory`
--   ADD PRIMARY KEY (`id`);

-- --
-- -- AUTO_INCREMENT for dumped tables
-- --

-- --
-- -- AUTO_INCREMENT for table `inventory_itemcategory`
-- --
-- ALTER TABLE `inventory_itemcategory`
--   MODIFY `id` bigint(20) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=9;
-- COMMIT;

-- /*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
-- /*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
-- /*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
