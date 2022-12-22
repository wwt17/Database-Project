import argparse
from pathlib import Path
import mysql.connector
from mysql.connector import errorcode
import azure
from azure.storage.blob import BlobServiceClient, ContainerClient


account_url = "https://wwt.blob.core.windows.net/"
credential = "oyo/RGj21u5Kgw/TD2f7GjBjyIHXk5OVApoLGsjmw8RJrZqdI/6tgEjCB6jzuul+bnGu/WWD6KKg+AStWyzg4Q=="

mysql_server_config = {
    'host': 'wwt-sql.mysql.database.azure.com',
    'user': 'wwt@wwt-sql',
    'password': None,
    'database': 'db',
    'client_flags': [mysql.connector.ClientFlag.SSL],
    'ssl_ca': './DigiCertGlobalRootG2.crt.pem'
}

def get_blob_service_client():
    service = BlobServiceClient(account_url=account_url, credential=credential)
    return service


def get_blob_container_client(container_name="container"):
    container_client = ContainerClient(account_url, container_name, credential=credential)
    try:
        container_client.create_container()
    except azure.core.exceptions.ResourceExistsError:
        pass
    return container_client


def get_mysql_connector():
    if mysql_server_config['password'] is None:
        mysql_server_config['password'] = input('mysql password: ')

    try:
        conn = mysql.connector.connect(**mysql_server_config)
        print("Connection established")
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with the user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
        raise
    else:
        return conn


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--type', choices=['blob', 'mysql'], default='blob')
    argparser.add_argument('--op', choices=['upload', 'download', 'insert', 'read', 'update', 'delete'], default='insert')
    argparser.add_argument('--files', nargs='+', default=['data.csv'])
    args = argparser.parse_args()

    if args.type == 'blob':
        container_client = get_blob_container_client()

        print('blob container before operation:')
        blob_list = container_client.list_blobs()
        for blob in blob_list:
            print(blob.name)

        if args.op == 'upload':
            for filename in args.files:
                print(f'upload file {filename}')
                with open(filename, 'rb') as data:
                    try:
                        container_client.upload_blob(filename, data)
                    except azure.core.exceptions.ResourceExistsError:
                        pass

        elif args.op == 'download':
            for filename in args.files:
                print(f'download file {filename}')
                blob_data = container_client.download_blob(filename)
                with open(filename + '.download', 'wb') as data_file:
                    blob_data.readinto(data_file)

        else:
            raise Exception(f"Operation {args.op} is unknown")

        print('blob container after operation:')
        blob_list = container_client.list_blobs()
        for blob in blob_list:
            print(blob.name)

    if args.type == 'mysql':
        try:
            conn = get_mysql_connector()
        except mysql.connector.Error as err:
            pass

        else:
            cursor = conn.cursor()

            if args.op == 'insert':
                # Drop previous table of same name if one exists
                cursor.execute("DROP TABLE IF EXISTS inventory;")
                print("Finished dropping table (if existed).")

                # Create table
                cursor.execute("CREATE TABLE inventory (id serial PRIMARY KEY, name VARCHAR(50), quantity INTEGER);")
                print("Finished creating table.")

                # Insert some data into table
                cursor.execute("INSERT INTO inventory (name, quantity) VALUES (%s, %s);", ("banana", 150))
                print("Inserted",cursor.rowcount,"row(s) of data.")
                cursor.execute("INSERT INTO inventory (name, quantity) VALUES (%s, %s);", ("orange", 154))
                print("Inserted",cursor.rowcount,"row(s) of data.")
                cursor.execute("INSERT INTO inventory (name, quantity) VALUES (%s, %s);", ("apple", 100))
                print("Inserted",cursor.rowcount,"row(s) of data.")

            elif args.op == 'read':
                # Read data
                cursor.execute("SELECT * FROM inventory;")
                rows = cursor.fetchall()
                print("Read",cursor.rowcount,"row(s) of data.")

                # Print all rows
                for row in rows:
                    print(f"Data row = {row}")

            elif args.op == 'update':
                # Update a data row in the table
                cursor.execute("UPDATE inventory SET quantity = %s WHERE name = %s;", (300, "apple"))
                print("Updated",cursor.rowcount,"row(s) of data.")

            elif args.op == 'delete':
                # Delete a data row in the table
                cursor.execute("DELETE FROM inventory WHERE name=%(param1)s;", {'param1':"orange"})
                print("Deleted",cursor.rowcount,"row(s) of data.")

            # Cleanup
            conn.commit()
            cursor.close()
            conn.close()
            print("Done.")
