import uuid
import json
import time
from database import DatabaseHandler
from whatsapp_handler import WhatsAppHandler
from payment_handler import PaymentHandler
from aws_handler import AWSHandler

class MessageProcessor:
    def __init__(self):
        self.db = DatabaseHandler()
        self.whatsapp = WhatsAppHandler()
        self.payment = PaymentHandler()
        self.aws = AWSHandler()
        self.processed_messages = {}
        self.help_message = """Hi! You can use this for creating HD version of images.
Commands:
• 'credits' - Check your credits
• 'buy' - Buy more credits
• Send image - Enhance to HD (4 credits)
For support: support@vinkasa.com"""
    
    def is_duplicate_message(self, message_id, from_number):
        current_time = time.time()
        key = f"{message_id}_{from_number}"
        
        if key in self.processed_messages:
            return True
        
        self.processed_messages[key] = current_time
        
        expired_keys = [k for k, t in self.processed_messages.items() if current_time - t > 300]
        for k in expired_keys:
            del self.processed_messages[k]
        
        return False
    
    def process_message(self, message):
        message_id = message.get('id')
        from_number = message.get('from')
        message_type = message.get('type')
        
        if not message_id or not from_number or not message_type:
            return
        
        if self.is_duplicate_message(message_id, from_number):
            return
        
        user = self.db.get_user(from_number)
        
        if message_type == 'text':
            self._process_text_message(from_number, message, user)
        elif message_type == 'image':
            self._process_image_message(from_number, message, user)
    
    def _process_text_message(self, from_number, message, user):
        text_body = message.get('text', {}).get('body', '').strip().lower()
        
        if not user:
            self.db.create_user(from_number)
            welcome_msg = f"Welcome! {self.help_message}\nYou get 4 free credits = 1 free image!\nMay we know your name?"
            self.whatsapp.send_message(from_number, welcome_msg)
            self.whatsapp.send_sample_images(from_number)
            return
        
        if text_body in ['hi', 'hello', 'help']:
            name = user.get('name', '')
            greeting = f"Hi {name}! " if name else "Hi! "
            self.whatsapp.send_message(from_number, f"{greeting}{self.help_message}")
        elif text_body == 'credits':
            credits = user.get('credits', 0)
            images = credits // 4
            self.whatsapp.send_message(from_number, f"💎 Credits: {credits}\n🖼️ Images: {images}")
        elif text_body == 'buy':
            self.payment.show_packages(from_number)
        elif text_body in ['1', '2', '3', '4']:
            self.payment.create_payment_link(from_number, text_body)
        elif not user.get('name') and len(text_body) > 1:
            self.db.update_user_name(from_number, text_body.title())
            self.whatsapp.send_message(from_number, f"Nice to meet you, {text_body.title()}! Send an image to enhance.")
        else:
            credits = user.get('credits', 0)
            images = credits // 4
            message_text = f"You have {credits} credits ({images} images). Send an image to enhance or buy more credits.\n\n{self.help_message}"
            self.whatsapp.send_message(from_number, message_text)
    
    def _process_image_message(self, from_number, message, user):
        if not user:
            self.db.create_user(from_number)
            self.whatsapp.send_message(from_number, f"Welcome! {self.help_message}\nProcessing your image...")
            self.whatsapp.send_sample_images(from_number)
            user = self.db.get_user(from_number)
        
        if not self.db.check_credits(from_number, 4):
            self.whatsapp.send_message(from_number, "❌ Insufficient credits (need 4). Send 'buy' to purchase more.")
            return
        
        image_info = message.get('image', {})
        image_id = image_info.get('id')
        
        if not image_id:
            self.whatsapp.send_message(from_number, "❌ Invalid image. Please try again.")
            return
        
        try:
            file_url = self.whatsapp.get_media_url(image_id)
            if not file_url:
                self.whatsapp.send_message(from_number, "❌ Could not access image. Please try again.")
                return
            
            unique_id = str(uuid.uuid4())
            input_filename = f"input_{unique_id}.jpg"
            output_filename = f"output_{unique_id}.jpg"
            
            uploaded = self.aws.upload_to_r2(file_url, input_filename)
            if not uploaded:
                self.whatsapp.send_message(from_number, "❌ Upload failed. Please try again.")
                return
            
            # SINGLE JOB CREATION POINT
            job_id = self.db.create_job(from_number, input_filename, output_filename)
            
            # Send single processing message
            self.whatsapp.send_message(from_number, "🔄 Processing your image... Please wait.")
            
            # Trigger instance management only
            self.aws.manage_instances()
            
            print(f"Job {job_id} created successfully for {from_number}")
                
        except Exception as e:
            print(f"Image processing error: {e}")
            self.whatsapp.send_message(from_number, "❌ Error processing image. Please try again.")
    
    def handle_webhook(self, event_body):
        """Handle webhook from spot instance after processing completion"""
        try:
            print(f"=== WEBHOOK PROCESSING START ===")
            print(f"Event body: {json.dumps(event_body, indent=2)}")
            
            output_file = event_body.get('output_file')
            processed_files = event_body.get('processed_files', [])
            
            files_to_process = [output_file] if output_file else processed_files
            
            for file_name in files_to_process:
                if not file_name:
                    continue
                    
                job = self.db.get_job_by_output_file(file_name)
                if not job or job['status'] == 'delivered':
                    continue
                
                user_phone = job['user_phone']
                job_id = job['id']
                
                self.db.update_job_status(job_id, 'completed')
                image_data = self.aws.download_from_r2(file_name)
                if not image_data:
                    self.whatsapp.send_message(user_phone, "❌ Failed to download processed image. Contact support.")
                    continue
                
                filename = f"HD_{file_name}"
                success = self.whatsapp.send_document(user_phone, image_data, filename, "✨ Your HD image is ready!")
                if success:
                    self.db.mark_job_as_delivered(job_id)
                    self.db.deduct_credits(user_phone, 4)
                    user = self.db.get_user(user_phone)
                    remaining = user.get('credits', 0)
                    imgs = remaining // 4
                    self.whatsapp.send_message(user_phone, f"✅ HD image delivered!\n💎 Credits: {remaining}\n🖼️ Images: {imgs}")
                else:
                    fallback = self.whatsapp.send_image(user_phone, image_data, "✨ Your HD image is ready!")
                    if fallback:
                        self.db.mark_job_as_delivered(job_id)
                        self.db.deduct_credits(user_phone, 4)
                        user = self.db.get_user(user_phone)
                        remaining = user.get('credits', 0)
                        imgs = remaining // 4
                        self.whatsapp.send_message(user_phone, f"✅ HD image delivered!\n💎 Credits: {remaining}\n🖼️ Images: {imgs}")
                    else:
                        self.whatsapp.send_message(user_phone, "❌ Failed to send HD image. Contact support at support@vinkasa.com")
            
            print(f"=== WEBHOOK PROCESSING COMPLETE ===")
        except Exception as e:
            print(f"❌ Webhook processing error: {e}")
            try:
                output_file = event_body.get('output_file')
                if output_file:
                    job = self.db.get_job_by_output_file(output_file)
                    if job:
                        self.whatsapp.send_message(job['user_phone'], "❌ Error processing your image. Please try again or contact support.")
            except:
                pass
    
    def cleanup_failed_jobs(self):
        """Cleanup jobs that have been processing for too long"""
        try:
            conn = self.db.get_connection()
            cur = conn.cursor()
            cur.execute("""
                UPDATE jobs
                SET status = 'failed'
                WHERE status IN ('pending','processing')
                  AND created_at < NOW() - INTERVAL '10 minutes'
            """)
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"Cleanup error: {e}")
