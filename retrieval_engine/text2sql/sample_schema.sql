-- Sample PostgreSQL Schema for E-commerce Database
-- This schema includes common tables for an e-commerce application
-- Use this to test the Text2SQL system

-- Drop tables if they exist (for clean setup)
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS categories;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS regions;

-- Create Regions table
CREATE TABLE regions (
    region_id SERIAL PRIMARY KEY,
    region_name VARCHAR(50) NOT NULL,
    country VARCHAR(50) NOT NULL
);

-- Create Customers table
CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    phone VARCHAR(20),
    address TEXT,
    city VARCHAR(50),
    region_id INTEGER REFERENCES regions(region_id),
    postal_code VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create Categories table
CREATE TABLE categories (
    category_id SERIAL PRIMARY KEY,
    category_name VARCHAR(50) NOT NULL,
    description TEXT
);

-- Create Products table
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    product_name VARCHAR(100) NOT NULL,
    description TEXT,
    category_id INTEGER REFERENCES categories(category_id),
    price DECIMAL(10, 2) NOT NULL,
    stock_quantity INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create Orders table
CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) CHECK (status IN ('pending', 'processing', 'shipped', 'delivered', 'cancelled')) DEFAULT 'pending',
    shipping_address TEXT,
    shipping_city VARCHAR(50),
    shipping_region_id INTEGER REFERENCES regions(region_id),
    shipping_postal_code VARCHAR(20),
    total_amount DECIMAL(12, 2),
    payment_method VARCHAR(50)
);

-- Create Order Items table
CREATE TABLE order_items (
    order_item_id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(order_id),
    product_id INTEGER REFERENCES products(product_id),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    unit_price DECIMAL(10, 2) NOT NULL,
    discount DECIMAL(5, 2) DEFAULT 0,
    subtotal DECIMAL(10, 2) GENERATED ALWAYS AS (quantity * unit_price * (1 - discount)) STORED
);

-- Insert sample data for Regions
INSERT INTO regions (region_name, country) VALUES
('North America', 'USA'),
('Europe', 'Germany'),
('Asia', 'Japan'),
('South America', 'Brazil'),
('Oceania', 'Australia');

-- Insert sample data for Customers
INSERT INTO customers (first_name, last_name, email, phone, address, city, region_id, postal_code) VALUES
('John', 'Doe', 'john.doe@example.com', '555-123-4567', '123 Main St', 'New York', 1, '10001'),
('Jane', 'Smith', 'jane.smith@example.com', '555-987-6543', '456 Elm St', 'Berlin', 2, '10115'),
('Akira', 'Tanaka', 'akira.tanaka@example.com', '555-555-5555', '789 Sakura Ave', 'Tokyo', 3, '100-0001'),
('Carlos', 'Rodriguez', 'carlos.rodriguez@example.com', '555-111-2222', '101 Plaza Major', 'São Paulo', 4, '01310-100'),
('Emma', 'Wilson', 'emma.wilson@example.com', '555-444-3333', '202 Beach Rd', 'Sydney', 5, '2000');

-- Insert sample data for Categories
INSERT INTO categories (category_name, description) VALUES
('Electronics', 'Electronic devices and gadgets'),
('Clothing', 'Apparel and fashion items'),
('Books', 'Books and literature'),
('Home & Kitchen', 'Household and kitchen items'),
('Sports & Outdoors', 'Sporting goods and outdoor equipment');

-- Insert sample data for Products
INSERT INTO products (product_name, description, category_id, price, stock_quantity) VALUES
('Laptop Pro X', '15-inch laptop with high performance specs', 1, 1299.99, 50),
('Smartphone Y', 'Latest smartphone with advanced camera', 1, 899.99, 100),
('Cotton T-Shirt', 'Comfortable cotton t-shirt in various colors', 2, 19.99, 200),
('Designer Jeans', 'Premium denim jeans with modern fit', 2, 59.99, 75),
('Fantasy Novel: The Lost Kingdom', 'Bestselling fantasy novel', 3, 24.99, 150),
('Cooking Masterclass', 'Comprehensive cookbook for all skill levels', 3, 34.99, 60),
('Blender Deluxe', 'High-powered blender for smoothies and more', 4, 79.99, 40),
('Bedding Set', 'Luxury bedding set with high thread count', 4, 129.99, 30),
('Tennis Racket', 'Professional grade tennis racket', 5, 149.99, 25),
('Hiking Backpack', 'Durable backpack for outdoor adventures', 5, 89.99, 35);

-- Insert sample data for Orders
INSERT INTO orders (customer_id, order_date, status, shipping_address, shipping_city, shipping_region_id, shipping_postal_code, total_amount, payment_method) VALUES
(1, '2023-01-15', 'delivered', '123 Main St', 'New York', 1, '10001', 1319.98, 'credit_card'),
(2, '2023-02-20', 'shipped', '456 Elm St', 'Berlin', 2, '10115', 959.98, 'paypal'),
(3, '2023-03-10', 'processing', '789 Sakura Ave', 'Tokyo', 3, '100-0001', 44.98, 'credit_card'),
(4, '2023-04-05', 'pending', '101 Plaza Major', 'São Paulo', 4, '01310-100', 269.97, 'bank_transfer'),
(1, '2023-05-12', 'delivered', '123 Main St', 'New York', 1, '10001', 239.98, 'credit_card'),
(5, '2023-06-18', 'shipped', '202 Beach Rd', 'Sydney', 5, '2000', 179.98, 'paypal'),
(2, '2023-07-22', 'delivered', '456 Elm St', 'Berlin', 2, '10115', 1299.99, 'credit_card'),
(3, '2023-08-30', 'processing', '789 Sakura Ave', 'Tokyo', 3, '100-0001', 129.99, 'credit_card');

-- Insert sample data for Order Items
INSERT INTO order_items (order_id, product_id, quantity, unit_price, discount) VALUES
(1, 1, 1, 1299.99, 0),
(1, 3, 1, 19.99, 0),
(2, 2, 1, 899.99, 0),
(2, 5, 2, 24.99, 0.10),
(3, 3, 1, 19.99, 0),
(3, 5, 1, 24.99, 0),
(4, 7, 1, 79.99, 0),
(4, 9, 1, 149.99, 0),
(4, 3, 2, 19.99, 0),
(5, 8, 1, 129.99, 0),
(5, 6, 2, 34.99, 0.20),
(6, 10, 2, 89.99, 0),
(7, 1, 1, 1299.99, 0),
(8, 8, 1, 129.99, 0);

-- Create an index on frequently queried columns
CREATE INDEX idx_customers_region ON customers(region_id);
CREATE INDEX idx_products_category ON products(category_id);
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_order_items_order ON order_items(order_id);
CREATE INDEX idx_order_items_product ON order_items(product_id);

-- Create a view for order summary
CREATE VIEW order_summary AS
SELECT 
    o.order_id,
    c.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    o.order_date,
    o.status,
    COUNT(oi.order_item_id) AS total_items,
    SUM(oi.quantity) AS total_quantity,
    o.total_amount,
    r.region_name
FROM 
    orders o
JOIN 
    customers c ON o.customer_id = c.customer_id
JOIN 
    regions r ON o.shipping_region_id = r.region_id
JOIN 
    order_items oi ON o.order_id = oi.order_id
GROUP BY 
    o.order_id, c.customer_id, customer_name, o.order_date, o.status, o.total_amount, r.region_name;

-- Create a view for product sales analysis
CREATE VIEW product_sales AS
SELECT 
    p.product_id,
    p.product_name,
    c.category_name,
    SUM(oi.quantity) AS total_sold,
    SUM(oi.subtotal) AS total_revenue,
    AVG(oi.unit_price) AS average_price,
    COUNT(DISTINCT o.customer_id) AS unique_customers
FROM 
    products p
JOIN 
    categories c ON p.category_id = c.category_id
LEFT JOIN 
    order_items oi ON p.product_id = oi.product_id
LEFT JOIN 
    orders o ON oi.order_id = o.order_id
GROUP BY 
    p.product_id, p.product_name, c.category_name;

-- Create a view for customer purchase history
CREATE VIEW customer_purchase_history AS
SELECT 
    c.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    r.region_name,
    COUNT(DISTINCT o.order_id) AS total_orders,
    SUM(o.total_amount) AS total_spent,
    MAX(o.order_date) AS last_order_date,
    AVG(o.total_amount) AS average_order_value
FROM 
    customers c
JOIN 
    regions r ON c.region_id = r.region_id
LEFT JOIN 
    orders o ON c.customer_id = o.customer_id
GROUP BY 
    c.customer_id, customer_name, c.email, r.region_name; 