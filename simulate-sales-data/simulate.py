from faker import Faker
import random
import csv
from datetime import datetime

# Initialize Faker
fake = Faker()

# Define categories for bar products
CATEGORIES = ['Beer', 'Wine', 'Cocktail', 'Spirit', 'Soft Drink']
LOCALS = ["Storsalen", "Bodegaen", "Klubben", "Strossa", "Selskapssiden", "Knaus", "Edgar", "Lyche", "Daglighallen", "Rundhallen", "Vollan", "Skala", "Vuelie", "Sitatet"]
BAR_SALE_POINTS = ["Bodegaen", "Daglighallen"]

def generate_daily_sales():
    sales = []
    # Get list of product IDs first
    products = generate_products()
    product_ids = [p['id'] for p in products]
    
    # Simulate a day's worth of sales for each bar
    for bar in BAR_SALE_POINTS:
        # Generate different sales patterns for different times of day
        # Morning (11-16): Very few sales
        # Evening (16-20): Medium sales
        # Night (20-02): Peak sales
        
        time_slots = [
            (11, 16, 0.3),  # Morning - low volume
            (16, 20, 0.7),  # Evening - medium volume
            (20, 2, 1.0),   # Night - high volume
        ]
        
        for hour in range(11, 26):  # 11 AM to 2 AM next day
            real_hour = hour if hour < 24 else hour - 24
            
            # Determine time slot multiplier
            multiplier = 0.3  # default to morning multiplier
            for start, end, mult in time_slots:
                if start <= real_hour < end or (end < start and (real_hour >= start or real_hour < end)):
                    multiplier = mult
                    break
            
            # Generate 0-5 sales per hour using exponential distribution
            num_sales = min(5, int(random.expovariate(1.0) * multiplier * 3))
            
            for _ in range(num_sales):
                sale = {
                    'bar': bar,
                    'timestamp': f"2024-03-14 {real_hour:02d}:{random.randint(0, 59):02d}:00",
                    'product_id': random.choice(product_ids),  # Use random product ID from list
                    'quantity': max(1, int(random.expovariate(0.5))),  # Most sales are 1-2 items
                }
                sales.append(sale)
    
    return sales

def generate_products(num_products=20):
    products = []
    
    for _ in range(num_products):
        category = random.choice(CATEGORIES)
        
        if category == 'Beer':
            name = f"{fake.company()} {random.choice(['IPA', 'Lager', 'Stout', 'Pale Ale'])}"
        elif category == 'Wine':
            name = f"{fake.last_name()} {random.choice(['Cabernet', 'Merlot', 'Chardonnay', 'Pinot Noir'])}"
        elif category == 'Cocktail':
            name = f"{fake.city()} {random.choice(['Martini', 'Margarita', 'Mojito', 'Sour'])}"
        elif category == 'Spirit':
            name = f"{fake.company()} {random.choice(['Vodka', 'Gin', 'Whiskey', 'Rum'])}"
        else:  # Soft Drink
            name = f"{fake.company()} {random.choice(['Cola', 'Lemonade', 'Tonic', 'Soda'])}"
            
        # Create a human readable ID from the name and category
        product_id = f"{category.lower()}-{name.lower().replace(' ', '-')}"
            
        product = {
            'id': product_id,
            'name': name,
            'category': category,
            'price': round(random.uniform(4.0, 25.0), 2),
            'cost': round(random.uniform(1.0, 10.0), 2),
        }
        products.append(product)
    
    return products

def write_to_csv(products, sales):
    # Write products to CSV
    with open('bar_products.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'name', 'category', 'price', 'cost'])
        writer.writeheader()
        writer.writerows(products)
    
    # Write sales to CSV
    with open('bar_sales.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['bar', 'timestamp', 'product_id', 'quantity'])
        writer.writeheader()
        writer.writerows(sales)

if __name__ == "__main__":
    # Generate sample data
    products = generate_products()
    sales = generate_daily_sales()
    
    # Write to CSV files
    write_to_csv(products, sales)
    
    print(f"Generated {len(products)} products and {len(sales)} sales records.")
    print("Data has been written to 'bar_products.csv' and 'bar_sales.csv'")
