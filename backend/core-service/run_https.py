"""
HTTPS Flask Runner for LAN access
Usage: python run_https.py <cert_file> <key_file> [port]
"""
import sys
import ssl
from app import create_app

def main():
    if len(sys.argv) < 3:
        print("Usage: python run_https.py <cert_file> <key_file> [port]")
        sys.exit(1)
    
    cert_file = sys.argv[1]
    key_file = sys.argv[2]
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
    
    app = create_app()
    
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_file, key_file)
    
    print(f"Starting Flask HTTPS server on port {port}...")
    app.run(host='0.0.0.0', port=port, ssl_context=context, debug=False, use_reloader=False)

if __name__ == '__main__':
    main()
