import os

import sys

import requests
import json
from dotenv import load_dotenv

load_dotenv()
class WhatsAppHandler:
    def __init__(self):
        self.access_token = os.getenv('WHATSAPP_ACCESS_TOKEN')
        self.phone_number_id = os.getenv('WHATSAPP_PHONE_NUMBER_ID')
        self.base_url = f"https://graph.facebook.com/v17.0/{self.phone_number_id}"
    
    def send_message(self, to_number, message_text):
        """Send text message"""
        try:
            url = f"{self.base_url}/messages"
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            }
            payload = {
                'messaging_product': 'whatsapp',
                'to': to_number,
                'type': 'text',
                'text': {'body': message_text}
            }
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"Message send error: {e}")
            return False
    
    def send_document(self, to_number, document_data, filename, caption=""):
        """Send document (for HD images)"""
        try:
            # Upload media
            media_url = f"{self.base_url}/media"
            headers = {'Authorization': f'Bearer {self.access_token}'}
            
            files = {
                'file': (filename, document_data, 'image/jpeg'),
                'type': (None, 'image/jpeg'),
                'messaging_product': (None, 'whatsapp')
            }
            
            media_response = requests.post(media_url, headers=headers, files=files, timeout=30)
            
            if media_response.status_code == 200:
                media_id = media_response.json().get('id')
                
                # Send as document
                message_url = f"{self.base_url}/messages"
                payload = {
                    'messaging_product': 'whatsapp',
                    'to': to_number,
                    'type': 'document',
                    'document': {
                        'id': media_id,
                        'caption': caption,
                        'filename': filename
                    }
                }
                
                response = requests.post(
                    message_url, 
                    headers={
                        'Authorization': f'Bearer {self.access_token}', 
                        'Content-Type': 'application/json'
                    }, 
                    json=payload, 
                    timeout=30
                )
                return response.status_code == 200
            else:
                print(f"Media upload failed: {media_response.status_code}")
                return False
                
        except Exception as e:
            print(f"Document send error: {e}")
            return False
    
    def send_image(self, to_number, image_data, caption=""):
        """Send image (fallback method)"""
        try:
            media_url = f"{self.base_url}/media"
            headers = {'Authorization': f'Bearer {self.access_token}'}
            
            files = {
                'file': ('image.jpg', image_data, 'image/jpeg'),
                'type': (None, 'image/jpeg'),
                'messaging_product': (None, 'whatsapp')
            }
            
            media_response = requests.post(media_url, headers=headers, files=files, timeout=30)
            
            if media_response.status_code == 200:
                media_id = media_response.json().get('id')
                
                message_url = f"{self.base_url}/messages"
                payload = {
                    'messaging_product': 'whatsapp',
                    'to': to_number,
                    'type': 'image',
                    'image': {
                        'id': media_id,
                        'caption': caption
                    }
                }
                
                response = requests.post(
                    message_url, 
                    headers={
                        'Authorization': f'Bearer {self.access_token}', 
                        'Content-Type': 'application/json'
                    }, 
                    json=payload, 
                    timeout=30
                )
                return response.status_code == 200
            else:
                print(f"Image upload failed: {media_response.status_code}")
                return False
                
        except Exception as e:
            print(f"Image send error: {e}")
            return False
    
    def send_sample_images(self, to_number):
        """Send sample images"""
        sample_files = ['1.png', '2.png', '3.png']
        for filename in sample_files:
            if os.path.exists(f'/var/task/{filename}'):
                with open(f'/var/task/{filename}', 'rb') as f:
                    self.send_image(to_number, f.read(), "Sample HD image")
    
    def get_media_url(self, media_id):
        """Get media URL from WhatsApp"""
        try:
            response = requests.get(
                f"https://graph.facebook.com/v17.0/{media_id}",
                headers={'Authorization': f'Bearer {self.access_token}'}
            )
            
            if response.status_code == 200:
                return response.json().get('url')
            return None
            
        except Exception as e:
            print(f"Media URL error: {e}")
            return None