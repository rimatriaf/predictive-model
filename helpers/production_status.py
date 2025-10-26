import pandas as pd

def get_production_times(engine):
    query = """
        SELECT 
            tanggal AS tanggal, 
            total
        FROM tr_process_cpd
        WHERE tanggal >= NOW() - INTERVAL 12 HOUR
        ORDER BY tanggal ASC
    """
    df = pd.read_sql(query, engine)
    if df.empty:
        return {
            "production_start": None,
            "production_stop": None
        }

    production_start = None
    production_stop = None
    found_running = False

    for _, row in df.iterrows():
        if production_start is None and row['total'] > 0:
            production_start = row['tanggal']
        if row['total'] > 0:
            found_running = True
        if found_running and row['total'] == 0:
            production_stop = row['tanggal']
            break

    return {
        "production_start": production_start.strftime("%Y-%m-%dT%H:%M:%S") if production_start else None,
        "production_stop": production_stop.strftime("%Y-%m-%dT%H:%M:%S") if production_stop else None,
    }