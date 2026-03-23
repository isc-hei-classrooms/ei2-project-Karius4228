import certifi
from influxdb_client import InfluxDBClient


if __name__ == "__main__": 
    org = "HESSOVS"
    bucket = "MeteoSuisse"
    token = "ixOI8jiwG1nn6a2MaE1pGa8XCiIJ2rqEX6ZCnluhwAyeZcrT6FHoDgnQhNy5k0YmVrk7hZGPpvb_5aaA-ZxhIw=="
    client = InfluxDBClient(url="https://timeseries.hevs.ch", token=token, org=org,
                            ssl_ca_cert=certifi.where(), timeout=1000000)
    
    query = 'from(bucket:"' + bucket + '")\
    |> range(start: -3h, stop: now())\
    |> filter(fn: (r) => r["_measurement"] == "Air temperature 2m above ground (current value)")\
    |> filter(fn: (r) => r["Site"] == "Sion")'
    tables = client.query_api().query(org=org, query=query)
    times = []
    data = []
    for table in tables:
        for record in table.records:
            times.append(record["_time"])
            data.append(record["_value"]) 
    client.close()
