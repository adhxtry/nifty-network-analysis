"""
Quick launcher script for the Nifty Network Analysis web application.
"""

import sys
import webbrowser
from threading import Timer

def open_browser():
    """Open the default browser to the app URL."""
    webbrowser.open_new('http://localhost:8050')

if __name__ == '__main__':
    # Import and run the app
    from app import app

    print("=" * 60)
    print("Nifty Network Analysis Web Application")
    print("=" * 60)
    print("\n🚀 Starting server...")
    print("📊 Access the app at: http://localhost:8050")
    print("\n⚠️  Press Ctrl+C to stop the server\n")

    # Open browser after 1 second
    Timer(1, open_browser).start()

    # Run the app
    app.run_server(debug=True, port=8050)
