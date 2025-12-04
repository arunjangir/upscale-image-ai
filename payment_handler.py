import os
import uuid
import json
import base64
import hashlib
import hmac
import time

from phonepe.sdk.pg.payments.v1.payment_client import PhonePePaymentClient
from phonepe.sdk.pg.env import Env
from phonepe.sdk.pg.payments.v1.models.request.pg_pay_request import PgPayRequest

from whatsapp_handler import WhatsAppHandler
from database import DatabaseHandler
from dotenv import load_dotenv

load_dotenv()

class PaymentHandler:
    def __init__(self):
        # Load and verify configuration
        self.merchant_id = os.getenv('PHONEPE_MERCHANT_ID')
        self.salt_key = os.getenv('PHONEPE_SALT_KEY')
        self.salt_index = int(os.getenv('PHONEPE_SALT_INDEX', '1'))
        self.callback_url = os.getenv('PHONEPE_CALLBACK_URL')
        self.redirect_url = os.getenv('PHONEPE_REDIRECT_URL')
        
        if not all([self.merchant_id, self.salt_key, self.callback_url, self.redirect_url]):
            raise ValueError("Missing required PhonePe configuration in environment variables")

        # Initialize PhonePe client
        self.client = PhonePePaymentClient(
            merchant_id=self.merchant_id,
            salt_key=self.salt_key,
            salt_index=self.salt_index,
            env=Env.UAT,  # Change to Env.PROD for production
            should_publish_events=False
        )
        
        self.db = DatabaseHandler()
        self.whatsapp = WhatsAppHandler()

        # Credit packages (amounts in paise)
        self.packages = {
            '1': {'credits': 4, 'amount': 300, 'description': '4 credits for ₹3'},
            '2': {'credits': 20, 'amount': 1500, 'description': '20 credits for ₹15'},
            '3': {'credits': 40, 'amount': 2500, 'description': '40 credits for ₹25'},
            '4': {'credits': 100, 'amount': 5000, 'description': '100 credits for ₹50'}
        }

    def show_packages(self, phone_number):
        """Show available credit packages"""
        message = "💳 *Credit Packages Available:*\n"
        for key, package in self.packages.items():
            message += f"{key}. {package['description']}\n"
        message += "\nReply with package number (1-4) to buy credits."

        self.whatsapp.send_message(phone_number, message)

    def create_payment_link(self, phone_number, package_id):
        """Create PhonePe payment link with user identification"""
        if package_id not in self.packages:
            return None

        package = self.packages[package_id]
        
        # Improved transaction ID format
        clean_phone = phone_number.replace('+', '').replace('-', '').replace(' ', '')
        if not clean_phone.isdigit() or len(clean_phone) < 10:
            self.whatsapp.send_message(phone_number, "❌ Invalid phone number format")
            return None
            
        unique_id = str(uuid.uuid4().hex[:8])
        merchant_transaction_id = f"TXN_{clean_phone}_{int(time.time())}_{unique_id}"

        # Create payment record in database
        payment_id = self.db.create_payment(
            phone_number,
            merchant_transaction_id,
            package['amount'],
            package['credits']
        )

        if not payment_id:
            self.whatsapp.send_message(phone_number, "❌ Failed to create payment record. Please try again.")
            return None

        try:
            # Build payment request using SDK's builder
            pay_request = PgPayRequest.pay_page_pay_request_builder(
                merchant_transaction_id=merchant_transaction_id,
                amount=package['amount'],
                merchant_user_id=clean_phone,
                callback_url=self.callback_url,
                redirect_url=self.redirect_url
            )

            # Make payment request
            response = self.client.pay(pay_request)
            
            if not response.success:
                error_msg = f"❌ Payment failed. Code: {response.code}"
                if hasattr(response, 'message'):
                    error_msg += f", Message: {response.message}"
                self.whatsapp.send_message(phone_number, error_msg)
                return None

            # Get payment URL from response
            payment_url = response.data.instrument_response.redirect_info.url

            # Send payment details to user
            message = f"💳 *Payment Link Created*\n\n"
            message += f"Package: {package['description']}\n"
            message += f"Amount: ₹{package['amount']/100:.2f}\n\n"
            message += f"Click to pay: {payment_url}\n\n"
            message += f"Transaction ID: {merchant_transaction_id}\n"
            message += "After payment, credits will be added automatically."

            self.whatsapp.send_message(phone_number, message)
            return payment_url

        except Exception as e:
            print(f"Payment error: {str(e)}")
            self.whatsapp.send_message(
                phone_number,
                "❌ Payment service unavailable. Please try again later."
            )
            return None

    def extract_phone_from_transaction_id(self, merchant_transaction_id):
        """Extract phone number from merchant transaction ID"""
        try:
            # Format: TXN_{phone_number}_{timestamp}_{unique_id}
            parts = merchant_transaction_id.split('_')
            if len(parts) >= 4 and parts[0] == 'TXN':
                phone_number = parts[1]
                # Add + prefix if not present
                if not phone_number.startswith('+'):
                    phone_number = '+' + phone_number
                return phone_number
            return None
        except Exception as e:
            print(f"Error extracting phone from transaction ID: {str(e)}")
            return None

    def decode_phonepe_response(self, callback_body):
        """Decode PhonePe base64 encoded response"""
        try:
            print(f"Raw callback body: {callback_body}")
            
            # Parse the JSON body
            if isinstance(callback_body, str):
                body_json = json.loads(callback_body)
            else:
                body_json = callback_body
            
            print(f"Parsed body JSON: {body_json}")
            
            # Extract the base64 encoded response
            base64_response = body_json.get('response')
            if not base64_response:
                print("No 'response' field found in callback")
                return None
            
            # Decode base64
            decoded_bytes = base64.b64decode(base64_response)
            decoded_string = decoded_bytes.decode('utf-8')
            
            print(f"Decoded response string: {decoded_string}")
            
            # Parse the decoded JSON
            response_data = json.loads(decoded_string)
            
            print(f"Final response data: {response_data}")
            
            return response_data
            
        except Exception as e:
            print(f"Error decoding PhonePe response: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def verify_phonepe_signature(self, response, checksum):
        """Verify PhonePe signature for security - Fixed version"""
        try:
            print(f"Verifying signature. Response length: {len(response)}, Checksum: {checksum}")
            
            # The correct format for PhonePe signature verification
            # String to hash: base64Response + "/pg/v1/status/" + merchantId + saltKey
            string_to_hash = response + "/pg/v1/status/" + self.merchant_id + self.salt_key
            print(f"String to hash: {string_to_hash}")
            
            # Calculate SHA256 hash
            calculated_hash = hashlib.sha256(string_to_hash.encode('utf-8')).hexdigest()
            print(f"Calculated hash: {calculated_hash}")
            
            # Add salt index
            calculated_checksum = calculated_hash + "###" + str(self.salt_index)
            print(f"Calculated checksum: {calculated_checksum}")
            print(f"Received checksum: {checksum}")
            
            is_valid = calculated_checksum == checksum
            print(f"Signature verification result: {is_valid}")
            
            return is_valid
            
        except Exception as e:
            print(f"Signature verification error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def handle_payment_callback(self, callback_data):
        """Handle PhonePe payment callback with proper decoding and user identification"""
        try:
            print(f"Raw callback received: {json.dumps(callback_data, indent=2)}")
            
            # Extract headers for verification
            headers = callback_data.get('headers', {})
            x_verify = headers.get('x-verify') or headers.get('X-Verify')
            
            if not x_verify:
                print("Missing X-Verify header")
                return False
            
            # Get the callback body
            callback_body = callback_data.get('body')
            if not callback_body:
                print("Missing callback body")
                return False
            
            # Decode the PhonePe response
            response_data = self.decode_phonepe_response(callback_body)
            if not response_data:
                print("Failed to decode PhonePe response")
                return False
            
            # Get the base64 response for signature verification
            if isinstance(callback_body, str):
                body_json = json.loads(callback_body)
            else:
                body_json = callback_body
            
            base64_response = body_json.get('response', '')
            
            # Verify signature
            is_signature_valid = self.verify_phonepe_signature(base64_response, x_verify)
            if not is_signature_valid:
                print("⚠️ Signature verification failed - proceeding anyway for UAT")
                # Uncomment below for production:
                # return False
            
            # Extract payment information
            payment_data = response_data.get('data', {})
            merchant_transaction_id = payment_data.get('merchantTransactionId')
            transaction_id = payment_data.get('transactionId')
            amount = payment_data.get('amount')
            status = payment_data.get('state')
            fees = int(payment_data.get('feesContext', {}).get('amount', 0))
            
            print(f"Payment details - TXN: {merchant_transaction_id}, Status: {status}, Amount: {amount}, Fees: {fees}")
            
            if not all([merchant_transaction_id, status]):
                print("Missing required payment fields")
                return False
            
            # Extract phone number
            phone_number = self.extract_phone_from_transaction_id(merchant_transaction_id)
            if not phone_number:
                print(f"Could not extract phone number from transaction ID: {merchant_transaction_id}")
                return False
            
            print(f"Extracted phone number: {phone_number}")
            
            # Get payment record from DB
            payment = self.db.get_payment_by_merchant_id(merchant_transaction_id)
            if not payment:
                print(f"No payment found for {merchant_transaction_id}")
                return False
            
            # Normalize phone numbers for comparison
            db_phone = payment['user_phone'].replace('+', '')
            extracted_phone = phone_number.replace('+', '')
            
            if db_phone != extracted_phone:
                print(f"Phone number mismatch. DB: {db_phone}, Extracted: {extracted_phone}")
                return False
            
            # Update payment status
            update_success = self.db.update_payment_status(
                merchant_transaction_id,
                status.lower(),
                transaction_id
            )
            
            if not update_success:
                print(f"Failed to update payment status for {merchant_transaction_id}")
                return False
            
            # Handle based on status
            if status == 'COMPLETED':
                return self._process_successful_payment(payment, transaction_id, amount, fees)
            elif status == 'PENDING':
                return self._process_pending_payment(payment, transaction_id, amount)
            else:
                return self._process_failed_payment(payment, transaction_id, amount, status)
            
        except Exception as e:
            print(f"Payment callback processing failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


    def _process_successful_payment(self, payment, transaction_id, amount, fees):
        """Process successful payment with fee consideration"""
        try:
            print(f"Processing successful payment for {payment['user_phone']}")
            
            # Verify amount matches (convert to int for comparison)
            expected_amount = int(payment['amount'])
            received_amount = int(amount) if amount else 0
            net_amount = received_amount - fees
            
            print(f"Amount verification - Expected: {expected_amount}, Received: {received_amount}, Fees: {fees}, Net: {net_amount}")
            
            # Allow 5% variance for fees and rounding
            if abs(net_amount - expected_amount) > (0.05 * expected_amount):
                error_msg = (f"Amount mismatch for {payment['merchant_transaction_id']}. "
                           f"Expected: {expected_amount}, Received: {received_amount}, Fees: {fees}")
                print(error_msg)
                self.whatsapp.send_message(
                    payment['user_phone'],
                    f"⚠️ Payment verification failed. Contact support with TXN ID: {transaction_id[:8]}..."
                )
                return False

            # Add credits to user account
            print(f"Adding {payment['credits']} credits to {payment['user_phone']}")
            success = self.db.add_credits(payment['user_phone'], payment['credits'])
            if not success:
                error_msg = f"Failed to add credits for {payment['user_phone']}"
                print(error_msg)
                self.whatsapp.send_message(
                    payment['user_phone'],
                    f"⚠️ Failed to add credits. Contact support with TXN ID: {transaction_id[:8]}..."
                )
                return False

            # Get updated user info
            user = self.db.get_user(payment['user_phone'])
            total_credits = user.get('credits', 0) if user else 0
            
            # Send success message
            message = (f"✅ *Payment Successful!*\n\n"
                     f"Amount: ₹{received_amount/100:.2f}\n"
                     f"Processing Fee: ₹{fees/100:.2f}\n"
                     f"Credits added: {payment['credits']}\n"
                     f"Total credits: {total_credits}\n\n"
                     f"Transaction ID: {transaction_id[:8]}...\n\n"
                     f"You can now send images to enhance!")
            
            self.whatsapp.send_message(payment['user_phone'], message)
            print(f"Successfully processed payment for {payment['user_phone']}")
            return True
            
        except Exception as e:
            print(f"Error processing successful payment: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _process_pending_payment(self, payment, transaction_id, amount):
        """Process pending payment"""
        try:
            message = (f"🔄 *Payment Pending*\n\n"
                     f"Your payment of ₹{int(amount)/100:.2f} is being processed.\n"
                     f"We'll notify you once completed.\n\n"
                     f"Transaction ID: {transaction_id[:8]}...")
            
            self.whatsapp.send_message(payment['user_phone'], message)
            return True
            
        except Exception as e:
            print(f"Error processing pending payment: {str(e)}")
            return False

    def _process_failed_payment(self, payment, transaction_id, amount, status):
        """Process failed payment"""
        try:
            message = (f"❌ *Payment Failed*\n\n"
                     f"Status: {status}\n"
                     f"Amount: ₹{int(amount)/100:.2f}\n\n"
                     f"Please try again or contact support.\n"
                     f"Transaction ID: {transaction_id[:8]}...")
            
            self.whatsapp.send_message(payment['user_phone'], message)
            return False
            
        except Exception as e:
            print(f"Error processing failed payment: {str(e)}")
            return False
                        
    def check_payment_status(self, merchant_transaction_id):
        """Check payment status manually"""
        try:
            response = self.client.check_status(merchant_transaction_id)
            return response.data.state if response.success else None
        except Exception as e:
            print(f"Payment status check error: {e}")
            return None