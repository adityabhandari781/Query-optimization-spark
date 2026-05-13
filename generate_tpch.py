# generate_tpch.py  — run once, generates ~500MB of Parquet files
import duckdb, os

OUT = r"C:\Users\adity\Desktop\code\GitHub\Query-optimization-spark\data\tpch"
os.makedirs(OUT, exist_ok=True)

con = duckdb.connect()
con.execute("INSTALL tpch; LOAD tpch;")
con.execute("CALL dbgen(sf=1)")   # scale factor 1 = ~1GB raw, ~500MB Parquet

for table in ["orders","lineitem","customer","part","supplier","partsupp","nation","region"]:
    con.execute(f"""
        COPY (SELECT * FROM {table})
        TO '{OUT}/{table}.parquet'
        (FORMAT PARQUET, COMPRESSION SNAPPY)
    """)
    count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"  {table:<12} {count:>10,} rows → {OUT}/{table}.parquet")

con.close()
print("Done.")