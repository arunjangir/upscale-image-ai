import sys
import os
from dotenv import load_dotenv

import pg8000
from datetime import datetime

load_dotenv()


class DatabaseHandler:
    def __init__(self):
        self.host = os.getenv('DB_HOST')
        self.user = os.getenv('DB_USER')
        self.password = os.getenv('DB_PASSWORD')
        self.database = os.getenv('DB_NAME')
        self.port = int(os.getenv('DB_PORT', '6543'))
    
    def get_connection(self):
        """Return a pg8000 connection with autocommit OFF (explicit commits)."""
        conn = pg8000.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database,
            port=self.port
        )
        conn.autocommit = False          # make sure we control commits/rollbacks
        return conn
    def update_job_status_by_files(self, input_file, output_file, status):
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE jobs
            SET status = %s,
                completed_at = CASE
                    WHEN %s IN ('completed', 'failed') THEN CURRENT_TIMESTAMP
                    ELSE completed_at
                END
            WHERE input_file = %s AND output_file = %s
            """,
            (status, status, input_file, output_file)
        )
        conn.commit()
        cur.close()
        conn.close()

    
    def init_db(self):
        conn = self.get_connection()
        cur = conn.cursor()
        
        # Users table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                phone_number VARCHAR(50) UNIQUE NOT NULL,
                name VARCHAR(100),
                credits INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Jobs table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id SERIAL PRIMARY KEY,
                user_phone VARCHAR(50) NOT NULL,
                input_file VARCHAR(500),
                output_file VARCHAR(500),
                status VARCHAR(50) DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
        """)
        
        # Payments table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS payments (
                id SERIAL PRIMARY KEY,
                user_phone VARCHAR(50) NOT NULL,
                merchant_transaction_id VARCHAR(100) UNIQUE NOT NULL,
                amount INTEGER NOT NULL,
                credits INTEGER NOT NULL,
                status VARCHAR(50) DEFAULT 'pending',
                phonepe_transaction_id VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
        """)
        
        conn.commit()
        cur.close()
        conn.close()
    
    def get_user(self, phone_number):
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE phone_number = %s", (phone_number,))
        columns = [col[0] for col in cur.description]
        user = cur.fetchone()
        cur.close()
        conn.close()
        
        if user:
            return dict(zip(columns, user))
        return None
    
    def create_user(self, phone_number):
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (phone_number, credits) VALUES (%s, %s) RETURNING id",
            (phone_number, 4)  # 4 free credits = 1 free image
        )
        user_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return user_id
    
    def update_user_name(self, phone_number, name):
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET name = %s, updated_at = CURRENT_TIMESTAMP WHERE phone_number = %s",
            (name, phone_number)
        )
        conn.commit()
        cur.close()
        conn.close()
    
    def check_credits(self, phone_number, amount=4):
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute("SELECT credits FROM users WHERE phone_number = %s", (phone_number,))
        result = cur.fetchone()
        cur.close()
        conn.close()
        
        if result and result[0] >= amount:
            return True
        return False
    
    def deduct_credits(self, phone_number, amount=4):
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET credits = credits - %s, updated_at = CURRENT_TIMESTAMP WHERE phone_number = %s AND credits >= %s",
            (amount, phone_number, amount)
        )
        affected = cur.rowcount
        conn.commit()
        cur.close()
        conn.close()
        return affected > 0
    
    def add_credits(self, phone_number, amount):
        """Add credits to a user atomically, with verification."""
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            # Lock the row so two callbacks can’t add credits twice
            cur.execute(
                "SELECT id, credits FROM users WHERE phone_number = %s FOR UPDATE",
                (phone_number,)
            )
            row = cur.fetchone()
            if not row:
                print(f"User {phone_number} not found")
                conn.rollback()
                return False

            user_id, current_credits = row
            new_credits = current_credits + amount

            cur.execute(
                "UPDATE users SET credits = %s, updated_at = CURRENT_TIMESTAMP "
                "WHERE id = %s",
                (new_credits, user_id)
            )

            conn.commit()
            print(f"Successfully added {amount} credits → total {new_credits}")
            return True

        except Exception as e:
            conn.rollback()
            print(f"Error adding credits: {e}")
            import traceback; traceback.print_exc()
            return False
        finally:
            cur.close()
            conn.close()
    
    def create_job(self, user_phone, input_file, output_file):
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO jobs (user_phone, input_file, output_file, status) VALUES (%s, %s, %s, 'pending') RETURNING id",
            (user_phone, input_file, output_file)
        )
        job_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return job_id

    def mark_job_as_delivered(self, job_id):
        """Mark job as delivered to prevent duplicate processing"""
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute(
            "UPDATE jobs SET status = 'delivered', completed_at = CURRENT_TIMESTAMP WHERE id = %s AND status IN ('pending', 'processing', 'completed')",
            (job_id,)
        )
        affected = cur.rowcount
        conn.commit()
        cur.close()
        conn.close()
        return affected > 0

    def get_job_by_output_file(self, output_file):
        conn = self.get_connection()
        cur = conn.cursor()
        # Include 'completed' status to handle webhooks from spot instances
        cur.execute(
            "SELECT * FROM jobs WHERE output_file = %s AND status IN ('pending', 'processing', 'completed')",
            (output_file,)
        )
        columns = [col[0] for col in cur.description]
        job = cur.fetchone()
        cur.close()
        conn.close()
        
        if job:
            return dict(zip(columns, job))
        return None
    
    def update_job_status(self, job_id, status):
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute(
            "UPDATE jobs SET status = %s, completed_at = CURRENT_TIMESTAMP WHERE id = %s",
            (status, job_id)
        )
        conn.commit()
        cur.close()
        conn.close()
    
    def create_payment(self, user_phone, merchant_transaction_id, amount, credits):
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO payments (user_phone, merchant_transaction_id, amount, credits) VALUES (%s, %s, %s, %s) RETURNING id",
            (user_phone, merchant_transaction_id, amount, credits)
        )
        payment_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return payment_id
    
    def get_payment_by_merchant_id(self, merchant_transaction_id):
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM payments WHERE merchant_transaction_id = %s", (merchant_transaction_id,))
        columns = [col[0] for col in cur.description]
        payment = cur.fetchone()
        cur.close()
        conn.close()
        
        if payment:
            return dict(zip(columns, payment))
        return None
    
    def update_payment_status(self, merchant_transaction_id, status,
                            phonepe_transaction_id=None):
        """Update payment row only if status is actually changing."""
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            # Lock the row we’re about to update
            cur.execute(
                "SELECT status FROM payments WHERE merchant_transaction_id = %s FOR UPDATE",
                (merchant_transaction_id,)
            )
            row = cur.fetchone()
            if not row:
                print(f"No payment found for {merchant_transaction_id}")
                conn.rollback()
                return False

            current_status = row[0].lower()
            new_status = status.lower()

            if current_status == new_status:
                conn.rollback()           # nothing to change
                return True

            # Build the UPDATE dynamically to include phonepe_transaction_id if given
            if phonepe_transaction_id:
                cur.execute(
                    """
                    UPDATE payments
                    SET status = %s,
                        phonepe_transaction_id = %s,
                        completed_at = CASE WHEN %s IN ('completed', 'failed')
                                            THEN CURRENT_TIMESTAMP
                                            ELSE completed_at END
                    WHERE merchant_transaction_id = %s
                    """,
                    (new_status, phonepe_transaction_id, new_status,
                    merchant_transaction_id)
                )
            else:
                cur.execute(
                    """
                    UPDATE payments
                    SET status = %s,
                        completed_at = CASE WHEN %s IN ('completed', 'failed')
                                            THEN CURRENT_TIMESTAMP
                                            ELSE completed_at END
                    WHERE merchant_transaction_id = %s
                    """,
                    (new_status, new_status, merchant_transaction_id)
                )

            conn.commit()
            return cur.rowcount == 1

        except Exception as e:
            conn.rollback()
            print(f"Error updating payment status: {e}")
            import traceback; traceback.print_exc()
            return False
        finally:
            cur.close()
            conn.close()
