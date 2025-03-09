"""from google.cloud import bigquery
import os
import json

# Initialize BigQuery client
client = bigquery.Client()

def fetch_github_data():
    
    Fetches recent GitHub repository information from BigQuery.
    
    query = 
    SELECT repo_name, description, created_at, pushed_at, language
    FROM `bigquery-public-data.github_repos.repositories`
    WHERE created_at > '2023-01-01'
    ORDER BY pushed_at DESC
    LIMIT 100
    
    
    query_job = client.query(query)
    results = query_job.result()

    data = []
    for row in results:
        data.append({
            "repo_name": row.repo_name,
            "description": row.description,
            "created_at": row.created_at,
            "pushed_at": row.pushed_at,
            "language": row.language
        })

    with open("processed_data/github_data.json", "w") as f:
        json.dump(data, f, indent=4)

    print("✅ GitHub data extracted and saved.")

if __name__ == "__main__":
    fetch_github_data()

def fetch_google_trends():
    
    Fetches recent trending Google search queries from BigQuery.
    
    query = 
    SELECT term, week, country_name, score
    FROM `bigquery-public-data.google_trends.international_top_terms`
    WHERE week > '2023-01-01'
    ORDER BY score DESC
    LIMIT 100
    
    
    query_job = client.query(query)
    results = query_job.result()

    data = []
    for row in results:
        data.append({
            "term": row.term,
            "week": str(row.week),
            "country": row.country_name,
            "score": row.score
        })

    with open("processed_data/google_trends.json", "w") as f:
        json.dump(data, f, indent=4)

    print("✅ Google Trends data extracted and saved.")

if __name__ == "__main__":
    fetch_google_trends()

"""