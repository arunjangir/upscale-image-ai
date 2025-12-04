import json
import os
import sys
from database import DatabaseHandler
from message_processor import MessageProcessor
from payment_handler import PaymentHandler
from aws_handler import AWSHandler

def lambda_handler(event, context):
    try:
        db = DatabaseHandler()
        db.init_db()
        
        print(f"Event: {json.dumps(event, indent=2)}")
        
        headers = event.get('headers', {})
        x_verify = headers.get('x-verify') or headers.get('X-Verify')
        
        # Handle PhonePe Payment Callback
        if x_verify:
            print("=== PhonePe Payment Callback Detected ===")
            payment_handler = PaymentHandler()
            success = payment_handler.handle_payment_callback(event)
            
            if success:
                print("Payment callback processed successfully")
                return {
                    'statusCode': 200, 
                    'body': json.dumps({'status': 'success', 'message': 'Payment processed'})
                }
            else:
                print("Payment callback processing failed")
                return {
                    'statusCode': 400,
                    'body': json.dumps({'status': 'error', 'message': 'Payment processing failed'})
                }
        
        # Handle Gradio Webhook (from spot instance)
        webhook_body = None
        if 'body' in event:
            try:
                if isinstance(event['body'], str):
                    webhook_body = json.loads(event['body'])
                else:
                    webhook_body = event['body']
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                webhook_body = None
        
        # Check if this is a webhook from spot instance
        if webhook_body and webhook_body.get('status') == 'completed':
            print("=== Gradio Webhook Detected ===")
            print(f"Webhook body: {json.dumps(webhook_body, indent=2)}")
            processor = MessageProcessor()
            processor.handle_webhook(webhook_body)
            return {
                'statusCode': 200, 
                'body': json.dumps({'status': 'success', 'message': 'Webhook processed'})
            }
        
        # Also check direct event format (fallback)
        if event.get('status') == 'completed':
            print("=== Direct Webhook Detected ===")
            processor = MessageProcessor()
            processor.handle_webhook(event)
            return {
                'statusCode': 200, 
                'body': json.dumps({'status': 'success', 'message': 'Webhook processed'})
            }
        
        # Handle WhatsApp requests
        method = event.get('httpMethod') or event.get('requestContext', {}).get('http', {}).get('method')
        
        if method == 'GET':
            print("=== WhatsApp Verification Request ===")
            params = event.get('queryStringParameters') or {}
            hub_mode = params.get('hub.mode')
            hub_token = params.get('hub.verify_token')
            hub_challenge = params.get('hub.challenge')
            
            if hub_mode == 'subscribe' and hub_token == os.getenv('VERIFY_TOKEN', '12345'):
                print("WhatsApp verification successful")
                return {'statusCode': 200, 'body': hub_challenge}
            else:
                print("WhatsApp verification failed")
                return {'statusCode': 403, 'body': 'Forbidden'}
        
        elif method == 'POST':
            print("=== WhatsApp Message Received ===")
            try:
                if isinstance(event.get('body'), str):
                    body = json.loads(event.get('body', '{}'))
                else:
                    body = event.get('body', {})
                
                # Skip if this is a webhook (already handled above)
                if body.get('status') == 'completed':
                    print("Webhook already processed, skipping WhatsApp processing")
                    return {
                        'statusCode': 200, 
                        'body': json.dumps({'status': 'success', 'message': 'Webhook processed'})
                    }
                
                processor = MessageProcessor()
                
                if 'entry' in body:
                    for entry in body['entry']:
                        if 'changes' in entry:
                            for change in entry['changes']:
                                if change.get('field') == 'messages':
                                    value = change.get('value', {})
                                    
                                    if 'statuses' in value:
                                        print("Status update received, ignoring...")
                                        continue
                                    
                                    messages = value.get('messages', [])
                                    for message in messages:
                                        if message.get('type') in ['text', 'image']:
                                            print(f"Processing message: {message}")
                                            processor.process_message(message)
                
                return {
                    'statusCode': 200, 
                    'body': json.dumps({'status': 'success', 'message': 'WhatsApp message processed'})
                }
                
            except Exception as e:
                print(f"WhatsApp processing error: {e}")
                import traceback
                traceback.print_exc()
                return {
                    'statusCode': 500, 
                    'body': json.dumps({'error': f'WhatsApp processing error: {str(e)}'})
                }
        
        print("=== Unknown Request Type ===")
        print(f"Method: {method}, Headers: {headers}")
        return {
            'statusCode': 405, 
            'body': json.dumps({'error': 'Method not allowed'})
        }
        
    except Exception as e:
        print(f"Lambda error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'statusCode': 500, 
            'body': json.dumps({'error': f'Lambda error: {str(e)}'})
        }