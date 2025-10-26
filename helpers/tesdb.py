import mysql.connector

conn = mysql.connector.connect(
    host="10.10.2.25",
    port=6446,
    user="iot_prod",
    password="P@ssw0rd123",
    database="aio_iot_can"
)

print("Connected!" if conn.is_connected() else "Failed to connect")