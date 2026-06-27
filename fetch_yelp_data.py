import requests
import pandas as pd
import time

API_KEY = "XImYvanD-jK_jp9meNmfT7Idmt7_XALeBizYiDnGMVz06GUR4PVRKJlJxoBbhHbfiYytPDxoKKY3GNx7dbG_c3hvdmDr34AVfusDBz7ba9saCjkK7N2kxw_RKCE_anYx"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

url = "https://api.yelp.com/v3/businesses/search"
cities = ["Vancouver", "Burnaby", "New Westminster", "Surrey"]
all_stores = []

for city in cities:
    for page in range(3):
        offset_value = page * 50
        
        params = {
            "categories": "cafes,restaurants", 
            "location": f"{city}, BC",
            "limit": 50,
            "offset": offset_value
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                businesses = data.get("businesses", [])
                
                if not businesses:
                    break
                
                for biz in businesses:
                    coords = biz.get("coordinates", {})
                    all_stores.append({
                        "store_name": biz.get("name"),
                        "city": city,
                        "latitude": coords.get("latitude"),
                        "longitude": coords.get("longitude"),
                        "target_rating": biz.get("rating")
                    })
            else:
                break
                
        except Exception:
            break
            
        time.sleep(1)

df = pd.DataFrame(all_stores).drop_duplicates(subset=["latitude", "longitude"])
df['target_is_successful'] = (df['target_rating'] >= 4.0).astype(int)

df.to_csv("milestone1.csv", index=False)