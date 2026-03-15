from pymongo import MongoClient
from datetime import datetime

client = MongoClient("mongodb://localhost:27017/")
db = client["judgement_db"]

predictions_collection = db["predictions"]
users_collection = db["users"]

def save_prediction(text, case_name, court, judge, doc_type, acts, main_category, nature_of_dispute, source, confidence, subdivisions, user_id, court_decision, summary, embedding=None):
    if subdivisions is None:
        subdivisions = []
    data = {
        "text": text,
        "case_name": case_name,
        "court": court,
        "judge": judge,
        "doc_type": doc_type,
        "acts": acts,
        "main_category": main_category,
        "nature_of_dispute": nature_of_dispute,
        "subdivisions": subdivisions,
        "court_decision": court_decision,
        "summary": summary,
        "embedding": embedding,
        "source": source,
        "confidence": confidence,
        "timestamp": datetime.now(),
        "user_id": user_id
    }
    result = predictions_collection.insert_one(data)
    return result.inserted_id

def get_all_documents():
    return list(predictions_collection.find())