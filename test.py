from dotenv import load_dotenv
import os
from pymongo import MongoClient
from datetime import datetime

load_dotenv()

def get_notification_stats(db, start_time=None, end_time=None, project_id=None):
    query = {}
    if start_time and end_time:
        query["created_at"] = {"$gte": start_time, "$lt": end_time}
    if project_id:
        query["project_id"] = project_id
    
    pipeline = [
        {"$match": query},
        {
            "$group": {
                "_id": None,
                "total_requests": {"$sum": "$total_senders"},
                "successful_requests": {"$sum": "$successful_sends"},
                "failed_requests": {"$sum": "$failed_sends"}
            }
        }
    ]
    
    result = list(db.notifications.aggregate(pipeline))
    return result[0] if result else {"total_requests": 0, "successful_requests": 0, "failed_requests": 0}

# Kết nối MongoDB
client = MongoClient(os.getenv("MONGO_URI"))  # Thay thế bằng URI thực tế
db = client["edu_backend"]  # Thay thế bằng tên database

# Nhập input từ người dùng
input_date = input("Nhập ngày (YYYY-MM-DD HH:MM:SS): ")
input_project_id = input("Nhập project ID (hoặc để trống): ")

# Chuyển đổi input ngày thành datetime
start_time = datetime.strptime(input_date, "%Y-%m-%d %H:%M:%S")
end_time = start_time.replace(second=59)  # Giới hạn trong 1 phút

project_id = input_project_id if input_project_id else None

stats = get_notification_stats(db, start_time, end_time, project_id)
print(stats)
