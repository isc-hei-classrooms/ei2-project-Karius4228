import joblib
pred = joblib.load(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\ei2-project-Karius4228\01-data\models_saved\da_v4_predictions.joblib")
print(type(pred))
print(list(pred.keys()) if hasattr(pred, 'keys') else pred)