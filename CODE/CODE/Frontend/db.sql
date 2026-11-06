-- Create the database
CREATE DATABASE traffic;

-- Switch to the new database
USE traffic;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    age INT NOT NULL,
    gender ENUM('M', 'F') NOT NULL,
    mobile VARCHAR(10) NOT NULL
);
