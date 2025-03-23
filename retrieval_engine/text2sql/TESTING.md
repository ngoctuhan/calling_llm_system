# Testing the Text2SQL System

This guide provides instructions for setting up a test PostgreSQL database and running the Text2SQL example against it.

## Setting up the Test Database

### Prerequisites
- PostgreSQL installed and running
- `psql` command line tool available
- Basic knowledge of PostgreSQL

### Step 1: Create a Test Database

```bash
# Connect to PostgreSQL
psql -U postgres

# Create a new database
CREATE DATABASE ecommerce_test;

# Connect to the new database
\c ecommerce_test
```

### Step 2: Import the Sample Schema

```bash
# Exit psql if you're in it
\q

# Import the schema from the command line
psql -U postgres -d ecommerce_test -f retrieval_engine/text2sql/sample_schema.sql
```

## Configuring Environment Variables

Create or update your `.env` file with the test database connection details:

```
# PostgreSQL configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=ecommerce_test
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password

# OpenAI API configuration
OPENAI_API_KEY=your_openai_api_key

# Qdrant vector database configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

## Running the Example Script

Once the database is set up and your environment variables are configured, you can run the example script:

```bash
# Make sure your virtual environment is activated if you're using one
python -m retrieval_engine.text2sql.example
```

## Example Natural Language Queries to Try

Here are some example queries you can try with the Text2SQL system and the e-commerce database:

1. **Basic Queries**
   - "Show me all products in the Electronics category"
   - "List customers from Germany"
   - "What are the top 3 most expensive products?"

2. **Intermediate Queries**
   - "Show total sales for each product category"
   - "Which customers have placed more than 1 order?"
   - "Find all orders shipped to North America"
   - "Show me products with less than 50 items in stock"

3. **Advanced Queries**
   - "What is the average order value by region?"
   - "Show me the total revenue from each customer, sorted from highest to lowest"
   - "Find products that haven't been ordered in the last 3 months"
   - "Which category has the highest average product price?"
   - "Show me customers who have ordered products from at least 3 different categories"

4. **Analytical Queries**
   - "What percentage of total revenue comes from each region?"
   - "Show me the month-over-month sales growth in 2023"
   - "Which payment method is most popular among customers in Europe?"
   - "Find the average discount applied to orders by category"

## Troubleshooting

- **Connection Issues**: If you have trouble connecting to PostgreSQL, check your PostgreSQL service is running and that the credentials in your `.env` file are correct.
- **Missing Tables**: If tables are missing, ensure you imported the schema correctly.
- **API Key Issues**: For OpenAI API errors, verify your API key is valid and that you have sufficient credits.
- **Qdrant Connection**: If Qdrant connection fails, ensure the Qdrant server is running and accessible. 