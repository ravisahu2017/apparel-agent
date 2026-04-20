#add a method to fetch and update record
from tinydb import TinyDB, Query
from datetime import datetime

def get_db():
    return TinyDB('db/products.nogit.json')

def fetch_record(product_id):
    db = get_db()
    ProductQuery = Query()
    return db.search(ProductQuery.product_id == product_id)

def update_record(product_id, data):
    db = get_db()
    ProductQuery = Query()
    data["updated_at"] = datetime.now().isoformat()
    return db.update(data, ProductQuery.product_id == product_id)

def insert_record(data):
    db = get_db()
    data["created_at"] = datetime.now().isoformat()
    data["updated_at"] = datetime.now().isoformat()
    return db.insert(data)

