import sqlite3

conn = sqlite3.connect("database.db")  # make sure the path is correct
c = conn.cursor()

# List all tables
c.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables:", c.fetchall())

# Show all rows in the users table
c.execute("SELECT * FROM users;")
rows = c.fetchall()
for row in rows:
    print(row)

conn.close()
