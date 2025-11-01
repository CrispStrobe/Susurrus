#!/usr/bin/env python3
"""Susurrus - Main entry point"""
import logging
import sys

from PyQt6.QtWidgets import QApplication

# Configure logging before any imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
)

# import after logging is configured
from gui.main_window import MainWindow
from utils.dependency_check import check_ffmpeg_installation


def main():
    """Main application entry point"""
    logger = logging.getLogger(__name__)

    try:
        # Check FFMPEG before starting GUI
        check_ffmpeg_installation()

        # Create and run application
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()

        logger.info("Application started successfully")
        sys.exit(app.exec())

    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
