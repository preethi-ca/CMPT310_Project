import requests
import pandas as pd

API_KEY = "bCqWg6c1wQYS3DTCIpPKVJqkbuR9HnkGd5Lqj3tyWFNJ-8aUzeQ9ziE2dXgNWCII7RMh9lYLBNQpi5MfjBYDdhHmsdTpsosEIqdVeaOreVEwgvkBtJrJFprDa3NxanYx"

# restrict to these 6 cities
cities = ["Vancouver, BC", "Burnaby, BC", "New Westminster, BC", "Surrey, BC", "Coquitlam, BC", "Richmond, BC"]

# restrict to these 2 categories
categories = ["restaurants", "cafes"]

offsets = [0, 50, 100, 150, 200]

url = "https://api.yelp.com/v3/businesses/search"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

# a list that stores one dictionary per restaurant or cafe
businesses_data = []

for city in cities:
    for category in categories:
        for offset in offsets:

            # the last page has up to 40 results (240 total per search)
            if offset == 200:
                limit = 40
            else:
                limit = 50

            params = {
                "location": city,
                "categories": category,
                "limit": limit,
                "offset": offset
            }

            response = requests.get(url, headers=headers, params=params)

            data = response.json()

            # error handling
            if "businesses" not in data:
                print(city, category, offset, data)
                continue

            businesses = data["businesses"]

            for business in businesses:

                # only keep businesses that have a price level
                price = business.get("price")

                if price is None:
                    continue

                price_level = len(price)

                rating = business["rating"]

                # definition of success
                if rating >= 4.0:
                    is_successful = 1
                else:
                    is_successful = 0

                # some businesses may not have a category
                business_categories = business.get("categories", [])

                if len(business_categories) > 0:
                    primary_category = business_categories[0]["title"]
                else:
                    primary_category = None

                business_info = {
                    "yelp_id": business["id"],
                    "store_name": business["name"],
                    "city": business["location"]["city"],
                    "zip_code": business["location"]["zip_code"],
                    "latitude": business["coordinates"]["latitude"],
                    "longitude": business["coordinates"]["longitude"],
                    "review_count": business["review_count"],
                    "price_level": price_level,
                    "primary_category": primary_category,
                    "target_rating": rating,
                    "target_is_successful": is_successful
                }

                businesses_data.append(business_info)

businesses_df = pd.DataFrame(businesses_data)

# remove duplicate Yelp businesses
businesses_df = businesses_df.drop_duplicates(subset="yelp_id")

# keep only businesses that returned a city in one of the 6 restricted cities
businesses_df = businesses_df[businesses_df["city"].isin(["Vancouver", "Burnaby", "New Westminster", "Surrey", "Coquitlam", "Richmond"])]

# reset the row numbers
businesses_df = businesses_df.reset_index(drop=True)

# randomly keep 550 businesses
businesses_df = businesses_df.sample(
    n=min(550, len(businesses_df)),
    random_state=42
)

# reset the row numbers after sampling
businesses_df = businesses_df.reset_index(drop=True)

# remove the Yelp ID from the final dataset
businesses_df = businesses_df.drop(columns=["yelp_id"])

# save the dataframe as a CSV file
businesses_df.to_csv("data/price_level_update.csv", index=False)
