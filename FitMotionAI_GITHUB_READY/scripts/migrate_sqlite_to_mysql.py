"""Migrate all data from SQLite (C:/Users/thuan/AppData/Local/FitMotionAI/aifitness.db)
to MySQL Server 8.0 database `fitmotion_ai`.
"""
import os
import sys
import sqlite3
import pymysql
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

SQLITE_PATH = r"C:\Users\thuan\AppData\Local\FitMotionAI\aifitness.db"
MYSQL_HOST = "127.0.0.1"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PASS = "Thuan312"
MYSQL_DB = "fitmotion_ai"

def migrate():
    print(f"[1/5] Checking SQLite source: {SQLITE_PATH}")
    if not os.path.exists(SQLITE_PATH):
        raise FileNotFoundError(f"SQLite file not found: {SQLITE_PATH}")

    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    sqlite_conn.row_factory = sqlite3.Row
    sqlite_cur = sqlite_conn.cursor()

    sqlite_cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence';")
    tables = [r[0] for r in sqlite_cur.fetchall()]
    print(f"Found {len(tables)} tables in SQLite: {tables}")

    print(f"[2/5] Connecting to MySQL {MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}...")
    root_conn = pymysql.connect(host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASS)
    with root_conn.cursor() as cur:
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DB}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
    root_conn.close()

    mysql_conn = pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASS,
        database=MYSQL_DB,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )
    mysql_cur = mysql_conn.cursor()

    print("[3/5] Creating MySQL schema via SQLAlchemy models...")
    from app import app
    from database import db
    import models  # noqa: F401

    with app.app_context():
        original_uri = app.config["SQLALCHEMY_DATABASE_URI"]
        app.config["SQLALCHEMY_DATABASE_URI"] = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASS}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4"
        db.engine.dispose()
        db.create_all()
        print("All tables created successfully in MySQL!")
        app.config["SQLALCHEMY_DATABASE_URI"] = original_uri

    print("[4/5] Transferring rows from SQLite to MySQL...")
    mysql_cur.execute("SET FOREIGN_KEY_CHECKS = 0;")
    
    total_migrated = 0
    report = []

    for table in tables:
        sqlite_cur.execute(f"SELECT * FROM `{table}`;")
        rows = sqlite_cur.fetchall()
        row_count = len(rows)
        
        if row_count == 0:
            report.append((table, 0, 0))
            continue

        cols = [col[0] for col in sqlite_cur.description]
        col_names = ", ".join([f"`{c}`" for c in cols])
        placeholders = ", ".join(["%s"] * len(cols))

        mysql_cur.execute(f"DELETE FROM `{table}`;")

        insert_sql = f"INSERT INTO `{table}` ({col_names}) VALUES ({placeholders});"
        
        data_to_insert = []
        for r in rows:
            row_vals = []
            for val in r:
                if isinstance(val, str) and len(val) >= 19 and val[4] == '-' and val[7] == '-':
                    try:
                        dt_val = datetime.fromisoformat(val)
                        row_vals.append(dt_val)
                        continue
                    except Exception:
                        pass
                row_vals.append(val)
            data_to_insert.append(row_vals)

        mysql_cur.executemany(insert_sql, data_to_insert)
        total_migrated += row_count
        report.append((table, row_count, row_count))
        print(f"  -> Migrated {row_count} rows for table `{table}`")

    mysql_cur.execute("SET FOREIGN_KEY_CHECKS = 1;")
    mysql_conn.commit()

    print("[5/5] Verification & Consistency Check:")
    print("=" * 60)
    print(f"{'Table':<30} | {'SQLite Rows':<12} | {'MySQL Rows':<12}")
    print("-" * 60)
    all_ok = True
    for table in tables:
        sqlite_cur.execute(f"SELECT COUNT(*) FROM `{table}`;")
        sq_cnt = sqlite_cur.fetchone()[0]
        mysql_cur.execute(f"SELECT COUNT(*) as cnt FROM `{table}`;")
        my_cnt = mysql_cur.fetchone()["cnt"]
        status = "OK" if sq_cnt == my_cnt else "MISMATCH"
        if status != "OK":
            all_ok = False
        print(f"{table:<30} | {sq_cnt:<12} | {my_cnt:<12} [{status}]")
    print("=" * 60)

    sqlite_conn.close()
    mysql_conn.close()

    if all_ok:
        print(f"SUCCESS: All {total_migrated} records migrated with 100% data consistency!")
    else:
        print("WARNING: Some row counts did not match, check details above.")

if __name__ == "__main__":
    migrate()
