import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    """Start the NoPinocchio demo webapp."""
    parser = argparse.ArgumentParser(description="Start NoPinocchio Demo Webapp")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port to bind to (default: 7860)"
    )
    parser.add_argument(
        "--share", action="store_true", help="Create a public shareable link"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="URL of the NoPinocchio API (default: http://localhost:8000)",
    )

    args = parser.parse_args()
    os.environ["API_URL"] = args.api_url

    from nopin.demo.app import demo

    demo.launch(
        server_name=args.host, server_port=args.port, share=args.share, show_error=True
    )


if __name__ == "__main__":
    main()
